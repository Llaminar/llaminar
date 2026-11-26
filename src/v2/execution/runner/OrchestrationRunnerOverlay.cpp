/**
 * @file OrchestrationRunnerOverlay.cpp
 * @brief OrchestrationRunner MoE overlay residency authority, maintenance, sealing, and serving-family setup.
 */

#include "OrchestrationRunnerInternal.h"

namespace llaminar2
{
    bool OrchestrationRunner::freezeMoEExpertOverlayPlanForLoadedModel()
    {
        if (moe_pipeline_topology_)
            return freezeMoEPipelinePlanForLoadedModel();
        if (!model_ctx_ || !config_.moe_routed_expert_plan)
            return true;

        if (snapshot_capture_setup_.enabled() &&
            !snapshot_capture_setup_.memory_capacity.valid())
        {
            return setError(
                "Graph snapshot capture requires a complete pre-initialization device-memory capacity declaration");
        }

        const bool requested_without_placements =
            config_.moe_routed_expert_plan
                ->usesExpertOverlayAuthority() &&
            config_.moe_routed_expert_plan->placements.empty();

        if (retained_prepared_routed_weight_plan_)
        {
            if (!retained_prepared_weight_plan_validated_ ||
                !retained_prepared_weight_plan_ ||
                retained_prepared_weight_plan_->runtime.resident_graph_rows <=
                    0)
            {
                return setError(
                    "Retained ExpertOverlay physical plan lacks a validated "
                    "positive graph-capacity certificate");
            }
            if (!retained_expert_overlay_memory_admission_)
            {
                return setError(
                    "Retained ExpertOverlay physical plan lacks its immutable CPU/GPU memory-admission certificate");
            }

            /*
             * The requested identity already includes every declarative input
             * that can change automatic tier filling or its fixed graph/
             * migration BOM. Re-running capacity against live free-memory here
             * would charge the retained weights a second time and incorrectly
             * shrink or reject the same topology. Install the exact immutable
             * model-frozen quotas and placements instead; all runner-owned
             * graphs, streams, arenas, residency banks, and controllers are
             * still rebuilt below this boundary.
             */
            config_.moe_routed_expert_plan =
                std::make_shared<MoERoutedExpertPlacementPlan>(
                    *retained_prepared_routed_weight_plan_);
            moe_expert_overlay_memory_admission_ =
                retained_expert_overlay_memory_admission_;
            physical_memory_admission_ =
                moe_expert_overlay_memory_admission_
                    ->physical_memory_admission;
            if (!physical_memory_admission_)
            {
                return setError(
                    "Retained ExpertOverlay capacity plan lacks its topology-wide physical-memory authority");
            }
            if (snapshot_capture_setup_.enabled())
            {
                const int current_rank = mpi_ctx_ ? mpi_ctx_->rank()
                                                  : plan_.rank;
                for (const auto &bom :
                     physical_memory_admission_->plan().resources())
                {
                    if (bom.resource().world_rank != current_rank ||
                        !bom.resource().device.is_gpu())
                    {
                        continue;
                    }
                    if (bom.bytes(
                            PhysicalMemoryOwner::GraphSnapshotArena) <
                        snapshot_capture_setup_.memory_capacity
                            .per_accelerator_bytes)
                    {
                        return setError(
                            "Retained ExpertOverlay physical-memory authority does not cover the requested graph snapshot capacity on " +
                            bom.resource().device.toString());
                    }
                }
            }
            PerfStatsCollector::addCounter(
                "weight_loading",
                "retained_overlay_physical_plan_reuses",
                1.0,
                "model_setup",
                plan_.primary_device.toString(),
                {{"resident_graph_rows",
                  std::to_string(plan_.runtime.resident_graph_rows)},
                 {"prefill_segment_rows",
                  std::to_string(
                      plan_.runtime.moe_routed_prefill
                          .overlay_segment_rows)},
                 {"requested_placements",
                  requested_without_placements ? "automatic" : "explicit"}});
            return publishPhysicalMemoryAuthority();
        }

        try
        {
            const MTPRuntimeConfig &mtp = retainedMTPConfig();
            if (config_.moe_routed_expert_plan
                    ->usesExpertOverlayAuthority())
            {
                const auto overlay_world =
                    moe_expert_overlay_mpi_ctx_
                        ? moe_expert_overlay_mpi_ctx_
                        : mpi_ctx_;
                const int world_rank = overlay_world
                                           ? overlay_world->rank()
                                           : plan_.rank;
                const int world_size = overlay_world
                                           ? overlay_world->world_size()
                                           : 1;
                if (world_rank != plan_.rank)
                {
                    throw std::invalid_argument(
                        "ExpertOverlay capacity communicator rank differs from the production rank plan");
                }

                const auto metadata =
                    resolveMoERoutedExpertModelMetadataForModel(
                        *model_ctx_, mtp);
                const auto layer_weight_manifest =
                    buildMoEOverlayLayerWeightManifestFromLoader(model_ctx_->concreteLoader(),
                        metadata.num_layers,
                        metadata.num_experts);
                const auto execution_plan =
                    resolveMoEExpertOverlayExecutionPlan(
                        config_.moe_routed_expert_plan,
                        MoEExpertOverlayExecutionPlanResolverOptions{
                            .current_world_rank = world_rank,
                            .world_size = world_size,
                        });

                if (!prepareRankLocalGPUContexts(
                        execution_plan.currentRankPlan().local_devices,
                        "physical_capacity_resolution"))
                {
                    return false;
                }
                const auto policy = resolveMoEOverlayCapacityAdmissionPolicy(
                    *config_.moe_routed_expert_plan,
                    config_,
                    world_size,
                    metadata.num_layers,
                    metadata.num_experts);

                const auto &profile = planningModelMetadata().memoryProfile();
                const MoEOverlayGPUWeightLoadCapacityInput
                    gpu_weight_load{
                        .policy =
                            configuredGPUWeightLoadMemoryPolicy(),
                        .maximum_source_bytes =
                            maximumGGUFTensorPayloadBytes(
                                model_ctx_->concreteLoader().getModel()),
                    };

                /*
                 * Captured-prefill residency and expert residency spend the
                 * same physical bytes. Evaluate them as one descending
                 * admission search so a large graph bucket cannot consume the
                 * space required for complete fallback-tier coverage. All
                 * ranks execute the same candidate/consensus sequence before
                 * any graph or migration pool receives a stable address.
                 */
                std::vector<int> graph_row_candidates{0};
                if (debugEnv().execution.gpu_graphs &&
                    debugEnv().execution.prefill_graph_buckets)
                {
                    /*
                     * ExpertOverlay executes long prompts as ordered captured
                     * segments.  Its resident graph is therefore bounded by
                     * the configured physical segment, while max_seq_len
                     * continues to size the independent KV/request state.
                     * Admitting a larger dense graph here would make the
                     * workspace-family declaration build a non-executable
                     * full-context MoE graph and force its sparse scratch back
                     * to the very capacity this protocol is designed to avoid.
                     */
                    graph_row_candidates =
                        segmentedPrefillGraphRowCandidates(
                            debugEnv().execution.prefill_graph_bucket_sizes,
                            plan_.runtime.max_seq_len,
                            plan_.runtime.moe_routed_prefill
                                .overlay_segment_rows);
                    if (graph_row_candidates.empty())
                    {
                        throw std::invalid_argument(
                            "ExpertOverlay captured-prefill admission has no positive resident graph-row candidate");
                    }
                }
                requireIdenticalOverlayGraphRowCandidates(
                    graph_row_candidates, overlay_world);

                /*
                 * Resolve retained-family multiplicity with the same model
                 * manifest authority used later by activation preflight. The
                 * row scalar does not change which main/MTP families exist;
                 * it only has to satisfy the identity's positive geometry.
                 */
                const bool retains_mtp_capacity =
                    retainsMTPGraphCapacity(mtp);
                const int activation_max_decode_rows =
                    retains_mtp_capacity
                        ? std::max(
                              1,
                              resolveMTPRetainedTargetQueryRows(mtp))
                        : 1;
                const int activation_max_mtp_draft_depth =
                    resolveMTPRetainedDraftCapacity(mtp);
                const int activation_identity_rows = std::max(
                    activation_max_decode_rows,
                    std::max(
                        1,
                        plan_.runtime.moe_routed_prefill
                            .overlay_segment_rows));
                const auto activation_graph_family =
                    resolveMoEOverlayInferenceGraphFamilyIdentity(
                        model_ctx_->concreteLoader(),
                        model_ctx_->architecture(),
                        model_ctx_->totalBlockCount(),
                        retains_mtp_capacity
                            ? MoEOverlayMTPGraphFamilyPolicy::
                                  RetainModelSidecars
                            : MoEOverlayMTPGraphFamilyPolicy::MainOnly,
                        /*graph_family_generation=*/1,
                        activation_identity_rows,
                        activation_max_decode_rows,
                        std::max(1, plan_.runtime.batch_size),
                        activation_max_mtp_draft_depth);
                const std::size_t model_graph_topology_variant_count =
                    snapshot_capture_setup_.retainedModelGraphTopologyVariantCount();
                const MoEOverlayMemoryPlanInputRequest memory_request{
                    .model = profile,
                    .rank_plan = plan_,
                    .config = config_,
                    .inventory = cluster_inventory_,
                    .execution = execution_plan,
                    .capacity_policy = policy,
                    .retained_mtp = mtp,
                    .graph_family = activation_graph_family,
                    .prefill = {
                        .bucket_rows = debugEnv().execution.prefill_graph_bucket_sizes,
                        .minimum_sequence_rows = debugEnv().execution.prefill_graph_min_seq,
                        .maximum_cached_buckets = debugEnv().execution.prefill_graph_max_cached_buckets},
                    .gpu_weight_load = gpu_weight_load,
                    .snapshot_capacity = snapshot_capture_setup_.enabled()
                        ? snapshot_capture_setup_.memory_capacity : GraphSnapshotMemoryCapacity{},
                    .model_graph_topology_variant_count = model_graph_topology_variant_count,
                    .original_expert_manifest=layer_weight_manifest,
                    .weight_precision=WeightPrecision::NATIVE,
                    .native_source_placement=model_ctx_->concreteLoader().nativeWeightSourcePlacement()};

                /** A successful local admission retains the exact inputs it priced. */
                struct LocalCapacityCandidate
                {
                    MoEOverlayMemoryPlanInputs inputs;
                    MoEOverlayLocalCapacityPlannerResult capacity;
                };

                // Reuse the same authenticated phase protocol as metadata and
                // graph initialization. A failed local BOM may reject a shape;
                // a broken/mismatched collective must terminate this search.
                const auto capacity_phase = [&](RankInitializationPhaseIdentity identity, auto &&work)
                {
                    auto result = RankInitializationLifecycle::execute(identity, work,
                        [&](auto phase, auto local_outcome) -> RankInitializationConsensusResult
                        {
                            if (!overlay_world || overlay_world->world_size() <= 1)
                                return {.outcome = local_outcome == RankInitializationLocalOutcome::Succeeded
                                    ? RankInitializationConsensusOutcome::AllRanksSucceeded
                                    : RankInitializationConsensusOutcome::AtLeastOneRankFailed};
                            return MPIRankInitializationConsensus::reach(
                                overlay_world->communicator(), phase, local_outcome);
                        });
                    switch (result.status)
                    {
                    case RankInitializationPhaseStatus::Succeeded:
                    case RankInitializationPhaseStatus::LocalStepReturnedFailure:
                    case RankInitializationPhaseStatus::LocalStepThrewException:
                    case RankInitializationPhaseStatus::PeerStepFailed:
                        return result;
                    default:
                        throw std::runtime_error(result.diagnostic("ExpertOverlay capacity"));
                    }
                };

                std::optional<MoEOverlayLocalCapacityPlannerResult>
                    selected_local_capacity;
                std::optional<MoEOverlayCapacityDraft> selected_capacity_draft;
                std::optional<MoEOverlayResolvedCapacityPlan> selected_capacity;
                std::size_t selected_model_graph_identity_count = 0u;
                int selected_prefill_segment_rows = 0;
                std::string last_capacity_error;
                const auto host_service_projection=RetainedHostServiceBudgetProjection::original();
                for (const int candidate_rows : graph_row_candidates)
                {
                    const int candidate_resident_rows =
                        candidate_rows > 0
                            ? candidate_rows
                            : resolveActivationBufferSeqLen(
                                  plan_.runtime.max_seq_len,
                                  plan_.primary_device.toLocalDeviceId());
                    std::optional<LocalCapacityCandidate> local_candidate;
                    const auto local_result = capacity_phase(
                        {static_cast<std::uint32_t>(candidate_rows), "overlay-candidate.fixed-bom"}, [&]
                    {
                        // Keep fallible BOM assembly inside the same readiness
                        // vote as local admission. No peer may leave this epoch
                        // while another rank enters its budget collective.
                        auto inputs = buildMoEOverlayMemoryPlanInputs(
                            memory_request, candidate_resident_rows);
                        inputs.local_capacity.retained_host_services=host_service_projection;
                        auto capacity = MoEOverlayLocalCapacityPlanner::plan(inputs.local_capacity);
                        local_candidate.emplace(LocalCapacityCandidate{
                            std::move(inputs), std::move(capacity)});
                        return true;
                    });

                    if (!local_result.succeeded())
                    {
                        last_capacity_error = local_result.diagnostic("ExpertOverlay capacity");
                        LOG_DEBUG(
                            "[MoEOverlayCapacity] Rejected resident graph candidate "
                            << candidate_rows << " rows: "
                            << last_capacity_error);
                        continue;
                    }

                    std::optional<MoEOverlayCapacityDraft> candidate_capacity;
                    const auto complete_result = capacity_phase(
                        {static_cast<std::uint32_t>(candidate_rows), "overlay-candidate.complete-bom"}, [&]
                    {
                        const auto physical_budgets =
                            gatherOverlayCapacityBudgets(
                                local_candidate->capacity.physical_budgets,
                                overlay_world);
                        candidate_capacity =
                            MoEOverlayCapacityAdmission::resolveCapacityDraft(
                                *config_.moe_routed_expert_plan,
                                metadata.num_experts,
                                layer_weight_manifest,
                                physical_budgets,
                                policy);
                        return true;
                    });

                    if (!complete_result.succeeded())
                    {
                        last_capacity_error = complete_result.diagnostic("ExpertOverlay capacity");
                        LOG_DEBUG(
                            "[MoEOverlayCapacity] Rejected resident graph candidate "
                            << candidate_rows << " rows: "
                            << last_capacity_error);
                        continue;
                    }

                    // Explicit native head ranges are mandatory storage, so a
                    // row candidate is complete only when those same ranges fit.
                    // Automatic placement remains below frozen rows and quotas.
                    if (plan_.runtime.ngram_table_policy.isFixedHeadSharded())
                    {
                        std::optional<ImmutableRowTableAdmissionSendFrame> local_frame;
                        const auto inputs_result = capacity_phase(
                            {static_cast<std::uint32_t>(candidate_rows), "overlay-candidate.fixed-head-inputs"}, [&]
                        {
                            const std::vector<std::vector<DevicePlanConfig>> groups{local_candidate->capacity.device_inputs};
                            const auto upload = MemoryPlanner::immutableRowTableUploadBindings(profile, groups);
                            const auto request = MemoryPlanner::immutableRowTableAdmissionRequest(profile,
                                plan_.runtime.ngram_table_policy, groups, upload,
                                ImmutableRowTableHostPlacement::fromObservedNUMANode(plan_.numa_node));
                            std::vector<ImmutableRowTableAdmissionRequest> local_requests;
                            if (request) local_requests.push_back(*request);
                            local_frame.emplace(ImmutableRowTableAdmissionSendFrame::prepare(local_requests,
                                overlay_world ? overlay_world->rank() : plan_.rank, &profile));
                            return true;
                        });
                        if (!inputs_result.succeeded())
                            throw std::runtime_error(inputs_result.diagnostic("Fixed-head row candidate metadata"));

                        std::optional<MoEOverlayResolvedCapacityPlan> complete_candidate;
                        std::string table_capacity_error;
                        const auto table_result = capacity_phase(
                            {static_cast<std::uint32_t>(candidate_rows), "overlay-candidate.fixed-head-complete-bom"}, [&]
                        {
                            const auto requests = gatherOverlayImmutableTableRequests(*local_frame, overlay_world, profile);
                            auto finalized = finalizeMoEOverlayRowCandidate(
                                plan_.runtime.ngram_table_policy, [&]
                            {
                                auto admitted = candidate_capacity->admit(requests);
                                if (admitted.size() != 1u)
                                    throw std::logic_error("Fixed-head row candidate lost its exact stage");
                                return std::move(admitted.front());
                            });
                            if (auto *rejected = std::get_if<MoEOverlayRowCapacityRejected>(&finalized))
                            {
                                table_capacity_error = std::move(rejected->diagnostic);
                                return false;
                            }
                            auto *admitted = std::get_if<MoEOverlayRowAdmitted<MoEOverlayResolvedCapacityPlan>>(&finalized);
                            if (!admitted)
                                throw std::logic_error("Mandatory fixed-head row finalization was deferred");
                            complete_candidate = std::move(admitted->value);
                            return true;
                        });
                        // Capacity rejection is a valid search outcome. Vote
                        // separately on structural errors so every peer fails
                        // fatally rather than advancing to another row epoch.
                        const auto validity = capacity_phase(
                            {static_cast<std::uint32_t>(candidate_rows), "overlay-candidate.fixed-head-validity"}, [&]
                        {
                            return table_result.status != RankInitializationPhaseStatus::LocalStepThrewException;
                        });
                        if (!validity.succeeded())
                            throw std::runtime_error(table_result.diagnostic("Invalid fixed-head row candidate"));
                        if (!table_result.succeeded())
                        {
                            last_capacity_error = table_result.diagnostic("Fixed-head complete row BOM");
                            if (!table_capacity_error.empty()) last_capacity_error += ": " + table_capacity_error;
                            LOG_DEBUG("[MoEOverlayCapacity] Rejected fixed-head resident graph candidate "
                                << candidate_rows << " rows: " << last_capacity_error);
                            continue;
                        }
                        // Reuse this one final certificate; no authority or
                        // physical backing was created for rejected candidates.
                        selected_capacity = std::move(complete_candidate);
                    }

                    selected_local_capacity = std::move(local_candidate->capacity);
                    selected_capacity_draft = std::move(candidate_capacity);
                    selected_model_graph_identity_count =
                        local_candidate->inputs.model_graph_identity_count;
                    selected_prefill_segment_rows =
                        local_candidate->inputs.prefill_segment_rows;
                    plan_.runtime.resident_graph_rows =
                        selected_local_capacity->resident_graph_rows;
                    break;
                }

                if (!selected_local_capacity || !selected_capacity_draft)
                {
                    throw std::runtime_error(
                        "ExpertOverlay has no resident captured-prefill shape with complete expert coverage under the physical BOM" +
                        (last_capacity_error.empty()
                             ? std::string{}
                             : ": " + last_capacity_error));
                }

                if (!selected_capacity)
                {
                    std::optional<ImmutableRowTableAdmissionSendFrame> local_table_frame;
                    const auto table_inputs=capacity_phase({0,"overlay.final-ngram-local-inputs"},[&] {
                        const std::vector<std::vector<DevicePlanConfig>> groups{selected_local_capacity->device_inputs};
                        const auto upload=MemoryPlanner::immutableRowTableUploadBindings(profile,groups);
                        const auto request=MemoryPlanner::immutableRowTableAdmissionRequest(profile,
                            plan_.runtime.ngram_table_policy,groups,upload,ImmutableRowTableHostPlacement::fromObservedNUMANode(plan_.numa_node));
                        std::vector<ImmutableRowTableAdmissionRequest> local_table_requests;
                        if(request)local_table_requests.push_back(*request);
                        local_table_frame.emplace(ImmutableRowTableAdmissionSendFrame::prepare(local_table_requests,
                            overlay_world ? overlay_world->rank() : plan_.rank, &profile));
                        return true;
                    });
                    if(!table_inputs.succeeded())
                        throw std::runtime_error(table_inputs.diagnostic("Final ngram local metadata admission"));
                    const auto table_result=capacity_phase({0,"overlay.final-ngram-admission"},[&] {
                        const auto complete_requests=gatherOverlayImmutableTableRequests(*local_table_frame,overlay_world,profile);
                        auto admitted=selected_capacity_draft->admit(complete_requests);
                        if(admitted.size()!=1u)throw std::logic_error("Standalone overlay finalization lost its exact stage");
                        selected_capacity=std::move(admitted.front());
                        return true;
                    });
                    if(!table_result.succeeded())
                        throw std::runtime_error(table_result.diagnostic("Final ngram admission after frozen rows and quotas"));
                }
                plan_.runtime.attention_participants = projectAdmittedQwenSparseAttentionPrograms(
                    profile, selected_local_capacity->device_inputs, plan_.rank);
                for (const auto &placement : selected_capacity->ngram_table_placements)
                    if (placement->worldRank() == plan_.rank)
                        plan_.runtime.ngram_table_placement = placement;

                if (selected_prefill_segment_rows <= 0)
                {
                    throw std::logic_error(
                        "ExpertOverlay capacity admission selected no positive prefill segment");
                }
                /*
                 * Publish the admitted prefill shape separately from the
                 * shared graph arena. A depth-15 MTP family may legitimately
                 * retain sixteen rows while this schedule retains only the
                 * configured one- and nine-row prefill executables.
                 */
                plan_.runtime.moe_routed_prefill.overlay_segment_rows =
                    selected_prefill_segment_rows;

                // Preserve the priced shape through retained-model reuse. Live
                // setup must not reconstruct a different host trace bound.
                selected_capacity->host_demand_memory =
                    selected_local_capacity->host_demand_memory;
                selected_capacity->cpu_serial_expert_storage =
                    MemoryPlanner::cpuSerialExpertStorageAdmissions(selected_local_capacity->device_inputs);
                plan_.runtime.cpu_serial_expert_storage = selected_capacity->cpu_serial_expert_storage;
                if(selected_local_capacity->device_inputs.empty())throw std::logic_error("Selected CPU service admission lost its original device inputs");
                plan_.runtime.retained_host_services=selected_local_capacity->device_inputs.front().retained_host_services;
                if(!plan_.runtime.retained_host_services)throw std::logic_error("Selected CPU service admission lost its original declaration");
                for(const auto& input:selected_local_capacity->device_inputs)
                    if(input.retained_host_services!=plan_.runtime.retained_host_services)throw std::logic_error("Selected CPU service admission mixed declaration sessions");
                {
                    auto publication = *selected_capacity;
                    publication.activation_batch_inventory =
                        selected_local_capacity->activation_batch_inventory;
                    publication.physical_gpu_lane_demand =
                        selected_local_capacity->physical_gpu_lane_demand;
                    moe_expert_overlay_memory_admission_ =
                        std::make_shared<const MoEOverlayResolvedCapacityPlan>(
                            std::move(publication));
                }
                // Publication owns the complete admitted geometry and physical
                // ledger. Quota installation and evidence must borrow that
                // immutable authority, never its moved-from setup envelope.
                const auto &capacity = *moe_expert_overlay_memory_admission_;
                physical_memory_admission_ =
                    moe_expert_overlay_memory_admission_
                        ->physical_memory_admission;
                if (!physical_memory_admission_)
                {
                    throw std::logic_error(
                        "ExpertOverlay capacity resolution returned no topology-wide physical-memory authority");
                }
                // Publish only the selected successful admission's actual inputs,
                // never speculative candidate BOMs or reconstructed intervals.
                if (profile.architecture == "qwen4exp" &&
                    PerfStatsCollector::isDomainEnabled("gpu_graph_inventory"))
                {
                    std::vector<DeviceId> members;
                    for (const auto &address : plan_.local_tp_devices)
                        members.push_back(address.toLocalDeviceId());
                    if (!members.empty())
                    {
                        const auto source = resolveProfileTPDeclaration(profile,
                            config_.moe_routed_expert_plan->domains.size() == 1 &&
                                plan_.usesLocalTP() && !plan_.usesLocalPP() && !plan_.usesGlobalTP() &&
                                !plan_.prev_rank && !plan_.next_rank && plan_.first_layer == 0 &&
                                (plan_.last_layer < 0 || plan_.last_layer == profile.mainLayerCount() - 1)
                                ? TensorParallelMainPlacement::FullMainRankLocal : TensorParallelMainPlacement::OtherDeclared,
                            plan_.runtime.mtp, config_.moe_routed_expert_plan->continuation_domain_spec.effectiveDensePolicy(), members);
                        if (source.coverage() == KVHeadCoveragePolicy::QueryInterval)
                            query_kv_geometry::recordInstalledAssignments(profile,
                                selected_local_capacity->device_inputs, *physical_memory_admission_);
                    }
                }
                LOG_DEBUG(
                    "[MoEOverlayCapacity] Selected rank-local fixed BOM at "
                    << selected_local_capacity->resident_graph_rows
                    << " resident rows:\n"
                    << selected_local_capacity->fixed_memory_plan.renderTable());
                auto capacity_bound = std::make_shared<
                    MoERoutedExpertPlacementPlan>(
                    MoEOverlayCapacityResolver::installResolvedQuotas(
                        *config_.moe_routed_expert_plan,
                        capacity));
                config_.moe_routed_expert_plan =
                    std::move(capacity_bound);
                if (capacity.replica_cache_capacity)
                {
                    LOG_INFO("[MoEOverlayCapacity] Replica cache admitted "
                             << capacity.replica_cache_capacity->admitted()
                             << '/' << capacity.replica_cache_capacity->requested()
                             << " experts per layer/participant after complete model and graph admission");
                }

                for (const auto &resource : capacity.physical_resources)
                {
                    const auto &memory = resource.bom();
                    const std::size_t staging_bytes =
                        memory.bytes(
                            PhysicalMemoryOwner::WeightLoadStaging) +
                        memory.bytes(
                            PhysicalMemoryOwner::ActivationTransportStaging) +
                        memory.bytes(
                            PhysicalMemoryOwner::ExpertMigrationStaging);
                    const std::size_t shadow_bytes = memory.bytes(
                        PhysicalMemoryOwner::ExpertShadowSlots);
                    const std::size_t live_expert_bytes = memory.bytes(
                        PhysicalMemoryOwner::RoutedExpertWeights);
                    const std::size_t fixed_incremental_bytes =
                        memory.incrementalBytes() - staging_bytes -
                        shadow_bytes - live_expert_bytes;
                    LOG_DEBUG(
                        "[MoEOverlayCapacity] resource="
                        << resource.resourceId()
                        << " device=" << resource.device().toString()
                        << " usable="
                        << memory.resource().admission_available_bytes
                        << " fixed=" << fixed_incremental_bytes
                        << " staging=" << staging_bytes
                        << " shadow=" << shadow_bytes
                        << " live_experts="
                        << live_expert_bytes
                        << " remaining=" << resource.remainingBytes());
                    PerfStatsCollector::addCounter(
                        "moe_overlay_capacity",
                        "physical_resource_admitted",
                        1.0,
                        "model_setup",
                        resource.device().toString(),
                        {{"resource_id", resource.resourceId()},
                         {"usable_bytes", std::to_string(
                              memory.resource()
                                  .admission_available_bytes)},
                         {"fixed_bytes", std::to_string(
                                             fixed_incremental_bytes)},
                         {"staging_bytes", std::to_string(
                                               staging_bytes)},
                         {"shadow_bytes", std::to_string(
                                              shadow_bytes)},
                         {"live_expert_bytes", std::to_string(
                                                   live_expert_bytes)},
                         {"remaining_bytes", std::to_string(
                                                 resource.remainingBytes())}});
                }
                PerfStatsCollector::addCounter(
                    "moe_overlay_capacity",
                    "exact_capacity_plan_installed",
                    1.0,
                    "model_setup",
                    "expert_overlay",
                    {{"tiers", std::to_string(capacity.tiers.size())},
                     {"physical_resources", std::to_string(
                                                capacity.physical_resources.size())},
                     {"layers", std::to_string(
                                    capacity.layer_footprints.size())},
                     {"migration_storage",
                      toString(policy.migration_storage)},
                     {"resident_graph_rows", std::to_string(
                                                   selected_local_capacity
                                                       ->resident_graph_rows)},
                     {"prefill_segment_rows", std::to_string(
                                                  selected_prefill_segment_rows)},
                     {"model_graph_identities_per_variant",
                      std::to_string(selected_model_graph_identity_count)},
                     {"model_graph_topology_variants",
                      std::to_string(model_graph_topology_variant_count)},
                     {"physical_model_graph_identities",
                      std::to_string(
                          selected_model_graph_identity_count *
                          model_graph_topology_variant_count)},
                     {"priority_only", "true"}});
            }

            auto frozen_plan = freezeMoEExpertOverlayPlanForModel(
                *model_ctx_,
                config_.moe_routed_expert_plan,
                mtp);
            if (!frozen_plan)
                return true;

            config_.moe_routed_expert_plan = std::move(frozen_plan);
            if (config_.moe_routed_expert_plan
                    ->usesExpertOverlayAuthority())
            {
                LOG_DEBUG("[OrchestrationRunner] MoE expert overlay plan frozen: placements="
                          << config_.moe_routed_expert_plan->placements.size()
                          << " routed_tiers=" << config_.moe_routed_expert_plan->routed_tiers.size()
                          << " domains=" << config_.moe_routed_expert_plan->domains.size()
                          << (requested_without_placements ? " (planned from model metadata)" : ""));
            }
        }
        catch (const std::exception &e)
        {
            return setError(std::string("Failed to freeze MoE expert overlay plan from model metadata: ") + e.what());
        }

        if (physical_memory_admission_)
            return publishPhysicalMemoryAuthority();
        return true;
    }

    bool OrchestrationRunner::freezeMoEPipelinePlanForLoadedModel()
    {
        try
        {
            if (!model_ctx_ || !moe_pipeline_topology_ || config_.moe_routed_expert_plan ||
                !mpi_ctx_ || mpi_ctx_->rank() != moe_pipeline_topology_->rankPlan().rank ||
                mpi_ctx_->world_size() != (moe_pipeline_topology_->globalPipeline()
                    ? moe_pipeline_topology_->globalPipeline()->world_size : 1))
                throw std::invalid_argument("MoE pipeline admission requires the exact local model and compiled stage topology");
            if (snapshot_capture_setup_.enabled() && !snapshot_capture_setup_.memory_capacity.valid())
                throw std::invalid_argument("MoE pipeline snapshots require admitted setup capacity");
            if (retained_prepared_weight_plan_)
            {
                if (!retained_prepared_weight_plan_validated_ || !moe_pipeline_prepared_plan_)
                    throw std::invalid_argument("Retained MoE pipeline has no validated stage placement authority");
                if (pipelineRoutedWeightAuthorityRequestIdentity(*moe_pipeline_topology_) != retained_routed_weight_authority_identity_)
                    throw std::invalid_argument("Retained MoE pipeline stage policy differs from the requested model authority");
                if (snapshot_capture_setup_.enabled())
                    for (const auto &bom : moe_pipeline_prepared_plan_->admission()->physicalAdmission()->plan().resources())
                        if (bom.resource().device.is_gpu() && bom.bytes(PhysicalMemoryOwner::GraphSnapshotArena) <
                            snapshot_capture_setup_.memory_capacity.per_accelerator_bytes)
                            throw std::invalid_argument("Retained MoE pipeline does not admit the requested snapshot capacity");
            }
            else
            {
                const auto devices = rankLocalGPUExecutionDevices(plan_, nullptr, mpi_ctx_);
                if (!prepareRankLocalGPUContexts(devices, "pipeline_physical_capacity_resolution"))
                    return false;
                const auto &execution = debugEnv().execution;
                if (!execution.gpu_graphs || !execution.prefill_graph_buckets)
                    throw std::invalid_argument("MoE pipeline GPU admission requires complete captured serving families");
                auto admission = std::make_shared<const AdmittedMoEPipelineMemory>(AdmittedMoEPipelineMemory::admit(
                    *moe_pipeline_topology_, planningModelMetadata(), model_ctx_->concreteLoader(), cluster_inventory_,
                    {.prefill = {.bucket_rows = execution.prefill_graph_bucket_sizes,
                         .minimum_sequence_rows = execution.prefill_graph_min_seq,
                         .maximum_cached_buckets = execution.prefill_graph_max_cached_buckets,
                         .maximum_cached_buckets_explicit = execution.prefill_graph_max_cached_buckets_explicit},
                     .gpu_weight_load = {.policy = configuredGPUWeightLoadMemoryPolicy(),
                         .maximum_source_bytes = maximumGGUFTensorPayloadBytes(model_ctx_->concreteLoader().getModel())},
                     .snapshot_capacity = snapshot_capture_setup_.enabled()
                         ? snapshot_capture_setup_.memory_capacity : GraphSnapshotMemoryCapacity{},
                     .model_graph_topology_variant_count = snapshot_capture_setup_.retainedModelGraphTopologyVariantCount(),
                     .retained_host_services = RetainedHostServiceBudgetProjection::original(),
                     .native_context_preparation = plan_.runtime.native_context_preparation,
                     .original_parent = mpi_ctx_}));
                moe_pipeline_prepared_plan_ = MoEPipelinePreparedPlan::create(std::move(admission), model_ctx_);
            }
            const auto &admission = *moe_pipeline_prepared_plan_->admission();
            physical_memory_admission_ = admission.physicalAdmission();
            admission.publishRuntimeCapacity(plan_.runtime);
            if (!publishPhysicalMemoryAuthority()) return false;
            PerfStatsCollector::addCounter("moe_overlay_capacity", "pipeline_capacity_plan_installed", 1.0,
                "model_setup", {}, {{"stages", std::to_string(admission.stages().size())},
                    {"retained", retained_prepared_weight_plan_ ? "true" : "false"}, {"single_authority", "true"}});
            return true;
        }
        catch (const std::exception &error)
        {
            return setError(std::string("Failed to admit model-owned MoE pipeline stages: ") + error.what());
        }
    }

    bool OrchestrationRunner::initializeContinuationRequestGroup()
    {
        if (continuation_request_group_)
            return setError("Continuation request membership is setup-only");
        const auto context = moe_expert_overlay_mpi_ctx_
                                 ? moe_expert_overlay_mpi_ctx_ : mpi_ctx_;
        if (!context)
            return true; // A genuinely process-local runner has no MPI payloads.

        try
        {
            const auto execution = resolveOverlayExecutionPlanForRunner(
                config_.moe_routed_expert_plan, context);
            std::vector<int> consumers;
            if (execution)
            {
                consumers = execution->continuationWorldRanks();
                if (!std::binary_search(consumers.begin(), consumers.end(),
                                        execution->continuation_root_rank))
                    return setError("ExpertOverlay command root is not a continuation owner");
            }
            else
            {
                // Preserve the existing TP/PP consumer policy. The command root
                // may be resolved later by the graph's terminal-stage authority.
                for (int rank = 0; rank < context->world_size(); ++rank)
                    consumers.push_back(rank);
            }
            continuation_request_group_ =
                std::make_unique<ContinuationRequestGroup>(context, std::move(consumers));
            PerfStatsCollector::addCounter(
                "prefix_cache", "coordination_scope_installed", 1.0, "model_setup", {},
                {{"scope", continuation_request_group_->scopeName()},
                 {"continuation_ranks",
                  std::to_string(continuation_request_group_->consumerCount())},
                 {"participant",
                  continuation_request_group_->role() ==
                          ContinuationRequestGroup::Role::TokenConsumer
                      ? "true" : "false"},
                 {"expert_only_ranks_excluded",
                  continuation_request_group_->consumerCount() <
                          static_cast<std::size_t>(context->world_size())
                      ? "true" : "false"}});
            return true;
        }
        catch (const std::exception &error)
        {
            return setError(std::string("Continuation request membership admission failed: ") +
                            error.what());
        }
    }

    bool OrchestrationRunner::initializeMoEExpertOverlayResidencyAuthority()
    {
        if(cpu_expert_shard_ctx_)return setError("CPU expert domain requires original terminal shutdown before replacement");
        shutdownMoEExpertOverlayResidencyMaintenance(
            MoEOverlayMaintenanceDrainIntent::ResetComposition);
        moe_overlay_device_controller_graph_service_.reset();
        moe_overlay_device_controller_fabric_.reset();
        moe_overlay_device_controller_topology_.reset();
        moe_expert_overlay_participant_residency_.reset();
        moe_expert_overlay_residency_authority_.reset();
        moe_expert_overlay_decode_histogram_.reset();
        moe_expert_overlay_interference_probe_.reset();
        moe_overlay_prefill_interference_owner_ =
            MoEOverlayPrefillInterferenceOwner::OrchestrationScope;

        if (moe_pipeline_topology_)
        {
            try
            {
                if (!moe_pipeline_prepared_plan_ || !model_ctx_ || !moe_pipeline_stage_bindings_.empty())
                    throw std::logic_error("MoE pipeline residency requires one admitted model and fresh runtime owners");
                moe_pipeline_stage_bindings_ = moe_pipeline_prepared_plan_->createRuntimeBindings(
                    model_ctx_->concreteLoader(), physical_memory_authority_, mpi_ctx_, *moe_pipeline_topology_);
                return true;
            }
            catch (const std::exception &error)
            {
                return setError(std::string("Failed to construct MoE pipeline stage runtime: ") + error.what());
            }
        }

        const auto &plan = config_.moe_routed_expert_plan;
        if (!model_ctx_ || !plan ||
            !plan->usesExpertOverlayAuthority())
            return true;

        try
        {
            const MTPRuntimeConfig &mtp = retainedMTPConfig();
            const auto metadata =
                resolveMoERoutedExpertModelMetadataForModel(
                    *model_ctx_,
                    mtp);
            if (!moe_expert_overlay_memory_admission_)
                throw std::logic_error("Expert runtime setup has no admitted physical capacity");
            const auto overlay_world =
                moe_expert_overlay_mpi_ctx_ ? moe_expert_overlay_mpi_ctx_ : mpi_ctx_;
            const int current_rank = overlay_world ? overlay_world->rank() : 0;
            const int world_size = overlay_world ? overlay_world->world_size() : 1;
            const auto capacity_policy = resolveMoEOverlayCapacityAdmissionPolicy(
                *plan, config_, world_size, metadata.num_layers, metadata.num_experts);
            // Main-model and PP stage owners use one checked constructor. Its
            // unpublished result retires as a unit if controller setup fails.
            const auto setup = MoEOverlayResidencySetup::create({
                .plan = *plan, .metadata = metadata, .loader = model_ctx_->concreteLoader(),
                .rebalance = config_.moe_rebalance, .storage_policy = capacity_policy,
                .capacity = *moe_expert_overlay_memory_admission_,
                .memory = physical_memory_authority_, .world_rank = current_rank, .world_size = world_size,
                .original_model=ModelContextId{reinterpret_cast<uint64_t>(model_ctx_.get())}});
            const auto &economy_layer_catalog = setup.authority()->economyLayerCatalog();
            const auto &owner_map = setup.authority()->snapshot()->originalOwnerMap();
            if (plan->authority_execution ==
                MoEOverlayAuthorityExecutionKind::DeviceResident)
            {
                const auto *const physical_topology =
                    mpi_ctx_ ? mpi_ctx_->topology() : nullptr;
                if (!physical_topology)
                {
                    return setError(
                        "All-GPU ExpertOverlay requires physical rank-to-node topology for its device-resident controller");
                }
                std::vector<int> world_rank_node_ids;
                world_rank_node_ids.reserve(
                    static_cast<std::size_t>(physical_topology->world_size()));
                for (int rank = 0; rank < physical_topology->world_size(); ++rank)
                {
                    world_rank_node_ids.push_back(
                        physical_topology->get_placement(rank).node_id);
                }
                auto controller_topology =
                    resolveMoEOverlayDeviceControllerTopology(
                        *plan,
                        owner_map,
                        {.world_rank_node_ids = world_rank_node_ids});
                moe_overlay_device_controller_topology_ =
                    std::make_shared<const MoEOverlayDeviceControllerTopology>(
                        std::move(controller_topology));

                PerfStatsCollector::addCounter(
                    "moe_overlay_controller",
                    "device_topology_resolved",
                    1.0,
                    "model_setup",
                    moe_overlay_device_controller_topology_
                        ->leader_device.toString(),
                    {{"authority_execution", "device_resident"},
                     {"controller_groups",
                      std::to_string(
                          moe_overlay_device_controller_topology_->groups.size())},
                     {"leader_participant",
                      std::to_string(
                          moe_overlay_device_controller_topology_
                              ->leader_participant_id)},
                     {"leader_world_rank",
                      std::to_string(
                          moe_overlay_device_controller_topology_
                              ->leader_world_rank)},
                     {"node_local_mapped_control",
                      moe_overlay_device_controller_topology_
                              ->usesNodeLocalMappedControl()
                          ? "true"
                          : "false"},
                     {"topology_fingerprint",
                      std::to_string(
                      moe_overlay_device_controller_topology_
                              ->topology_fingerprint)}});

                if (moe_overlay_device_controller_topology_
                        ->usesNodeLocalMappedControl())
                {
                    const auto controller_world =
                        moe_expert_overlay_mpi_ctx_
                            ? moe_expert_overlay_mpi_ctx_
                            : mpi_ctx_;
                    if (!controller_world)
                    {
                        return setError(
                            "Mapped all-GPU ExpertOverlay controller has no production MPI identity");
                    }

                    const int local_world_rank = controller_world->rank();
                    const bool owns_local_participant = std::any_of(
                        moe_overlay_device_controller_topology_
                            ->participants.begin(),
                        moe_overlay_device_controller_topology_
                            ->participants.end(),
                        [local_world_rank](const auto &participant)
                        {
                            return participant.world_rank ==
                                   local_world_rank;
                        });
                    if (owns_local_participant)
                    {
                        const auto controller_capacity_policy =
                            resolveMoEOverlayCapacityAdmissionPolicy(
                                *plan,
                                config_,
                                controller_world->world_size(),
                                metadata.num_layers,
                                metadata.num_experts);
                        const std::uint32_t command_capacity =
                            completeMoEOverlayDeviceCommandCapacity(
                                metadata.num_layers,
                                metadata.num_experts,
                                moe_overlay_device_controller_topology_
                                    ->participants.size());
                        std::vector<std::uint64_t>
                            payload_bytes_per_layer;
                        if (economy_layer_catalog)
                        {
                            payload_bytes_per_layer.reserve(
                                economy_layer_catalog
                                    ->completeExpertBytesPerLayer().size());
                            for (const std::size_t bytes :
                                 economy_layer_catalog
                                     ->completeExpertBytesPerLayer())
                            {
                                payload_bytes_per_layer.push_back(
                                    static_cast<std::uint64_t>(bytes));
                            }
                        }
                        std::vector<MoEOverlayRouteTarget> initial_route_targets;
                        initial_route_targets.reserve(static_cast<std::size_t>(metadata.num_layers) * metadata.num_experts);
                        for (int layer = 0; layer < metadata.num_layers; ++layer)
                            for (int expert = 0; expert < metadata.num_experts; ++expert)
                            {
                                const auto *placement = owner_map.placementFor(metadata.first_model_layer + layer, expert);
                                if (!placement)
                                    throw std::logic_error("Device controller restore declaration lost its original logical placement");
                                const auto target = placement->shard_cohort
                                    ? MoEOverlayRouteTarget::cohort(placement->tier_idx)
                                    : MoEOverlayRouteTarget::whole(placement->owner_participant);
                                if (!moe_overlay_device_controller_topology_->groupForTarget(target))
                                    throw std::logic_error("Device controller restore declaration names a foreign original target");
                                initial_route_targets.push_back(target);
                            }
                        moe_overlay_device_controller_fabric_ =
                            std::make_shared<
                                MoEOverlayNodeLocalDeviceControllerFabric>(
                                MoEOverlayNodeLocalDeviceControllerFabric::
                                    Config{
                                        .mpi_ctx = controller_world,
                                        .topology =
                                            moe_overlay_device_controller_topology_,
                                        .num_layers =
                                            static_cast<std::uint32_t>(
                                                metadata.num_layers),
                                        .num_experts =
                                            static_cast<std::uint32_t>(
                                                metadata.num_experts),
                                        .routed_experts_per_token =
                                            static_cast<std::uint32_t>(
                                                model_ctx_->concreteLoader()
                                                    .getInt(
                                                        model_ctx_->architecture() +
                                                            ".expert_used_count",
                                                        0)),
                                        .command_capacity = command_capacity,
                                        .initial_durable_epoch = 1u,
                                        .payload_bytes_per_layer =
                                            std::move(
                                                payload_bytes_per_layer),
                                        .initial_route_targets =
                                            std::move(
                                                initial_route_targets),
                                        .minimum_window_activations =
                                            static_cast<std::uint64_t>(
                                                std::max<std::uint64_t>(
                                                    1u,
                                                    config_.moe_rebalance
                                                        .dynamic_min_window_activations)),
                                        .maximum_cycles_per_wave =
                                            config_.moe_rebalance
                                                .resolvedMigrationCyclesPerWave(),
                                        .dynamic_imbalance_threshold_per_mille =
                                            config_.moe_rebalance
                                                .dynamic_imbalance_threshold_per_mille,
                                        .dynamic_minimum_improvement_per_mille =
                                            config_.moe_rebalance
                                                .dynamic_min_improvement_per_mille,
                                        .dynamic_maximum_cycles_per_layer =
                                            config_.moe_rebalance
                                                .dynamic_max_swaps_per_layer,
                                        .dynamic_maximum_commands_per_wave =
                                            config_.moe_rebalance
                                                .dynamic_max_plan_entries_per_wave,
                                        .first_model_layer = metadata.first_model_layer,
                                    });
                        PerfStatsCollector::addCounter(
                            "moe_overlay_controller",
                            "device_fabric_materialized",
                            1.0,
                            "model_setup",
                            moe_overlay_device_controller_topology_
                                ->leader_device.toString(),
                            {{"mapping_bytes",
                              std::to_string(
                                  moe_overlay_device_controller_fabric_
                                      ->layout()
                                      .mapping_bytes)},
                             {"command_capacity",
                              std::to_string(command_capacity)},
                             {"local_participants",
                              std::to_string(
                                  moe_overlay_device_controller_fabric_
                                      ->localParticipantIds()
                                      .size())},
                             {"imbalance_threshold_per_mille",
                              std::to_string(
                                  config_.moe_rebalance
                                      .dynamic_imbalance_threshold_per_mille)},
                             {"minimum_improvement_per_mille",
                              std::to_string(
                                  config_.moe_rebalance
                                      .dynamic_min_improvement_per_mille)},
                             {"maximum_cycles_per_layer",
                              std::to_string(
                                  config_.moe_rebalance
                                      .dynamic_max_swaps_per_layer)},
                             {"maximum_commands_per_wave",
                              std::to_string(
                                  config_.moe_rebalance
                                      .dynamic_max_plan_entries_per_wave)},
                             {"policy_owner", "device"},
                             {"host_policy_mirror", "false"}});
                    }
                }

                /*
                 * Multi-group lifecycle graphs are materialized only after
                 * every inference graph has sealed its persistent device
                 * resources. Authority selection remains frozen here; the
                 * maintenance initializer must prove the corresponding
                 * retained controller family before request admission.
                 */
            }
            moe_expert_overlay_decode_histogram_ = setup.histogram();
            moe_expert_overlay_residency_authority_ = setup.authority();
            moe_expert_overlay_participant_residency_ = setup.residency();
            moe_expert_overlay_interference_probe_ = setup.interference();
            if(!initializeCPUExpertShardDomain())return false;
        }
        catch (const std::exception &error)
        {
            return setError(
                std::string(
                    "Failed to initialize production ExpertOverlay residency authority: ") +
                error.what());
        }

        return true;
    }

    bool OrchestrationRunner::initializeCPUExpertShardDomain()
    {
        const auto snapshot=moe_expert_overlay_residency_authority_->snapshot();
        if(!snapshot || !snapshot->jointOwnerMap())return true;
        const auto &plan=*snapshot->placement_plan;
        // GPU joint source declarations have their own original native local
        // context. They must never enter the CPU domain/barrier owner.
        if (!MoEExpertOwnerMap::usesCPUJointFragments(plan)) return true;
        if(plan.authority_execution!=MoEOverlayAuthorityExecutionKind::HostResident)
            return setError("CPU expert domain requires its original declared host residency authority");
        const auto world=moe_expert_overlay_mpi_ctx_?moe_expert_overlay_mpi_ctx_:mpi_ctx_;
        if(!world || !physical_memory_authority_)return setError("CPU expert domain lost original world/PMA authority");
        std::optional<NativeCPUCollectiveProgram> program;
        std::optional<std::variant<NativeCPUCollectiveProgram::DomainUse,NativeCPUCollectiveProgram::ProcessUse>> use;
        const auto prepare=runRankSynchronizedInitializationPhase("CPU expert original domain geometry",
            {0,"cpu_expert_original_domain_geometry"},InitializationConsensusScope::ExpertOverlayContext,[&] {
                int domain_ordinal=-1;
                for(size_t d=0;d<plan.domains.size();++d) {
                    const auto &candidate=plan.domains[d];
                    if(candidate.routed_compute_policy!=RoutedExpertComputePolicy::TensorSharded || candidate.participants.size()<=1 ||
                       !std::all_of(candidate.participants.begin(),candidate.participants.end(),[](const auto& p){return p.isCPU();}))continue;
                    if(domain_ordinal!=-1)throw std::invalid_argument("CPU joint serving currently requires one original expert domain");
                    domain_ordinal=static_cast<int>(d);
                }
                if(domain_ordinal<0)throw std::logic_error("Original CPU joint source has no declared domain");
                const auto &domain=plan.domains[static_cast<size_t>(domain_ordinal)];
                std::shared_ptr<const MoEExpertShardCohort> cohort;
                for(const auto &row:snapshot->originalOwnerMap().owners())if(row.domain_name==domain.name && row.shard_cohort) {
                    if(!cohort)cohort=row.shard_cohort;
                    else if(cohort->participants().size()!=row.shard_cohort->participants().size() ||
                        !std::equal(cohort->participants().begin(),cohort->participants().end(),row.shard_cohort->participants().begin()))
                        throw std::invalid_argument("CPU expert source layers changed original domain membership");
                }
                if(!cohort || domain.scope!=ExecutionDomainScope::NODE_LOCAL)
                    throw std::invalid_argument("CPU joint serving requires positive original native fragments and NODE_LOCAL membership");
                std::vector<NativeCPUCollectiveProgram::Participant> members;
                int original_node=-1;
                for(const auto &member:cohort->participants()) {
                    if(member.world_rank<0 || static_cast<size_t>(member.world_rank)>=cluster_inventory_.ranks.size())
                        throw std::invalid_argument("CPU expert source contains foreign original rank");
                    const auto &rank=cluster_inventory_.ranks[static_cast<size_t>(member.world_rank)];
                    if(rank.rank!=member.world_rank || rank.cpu.numa_node!=member.address.numa_node ||
                       (original_node>=0 && rank.node_id!=original_node))
                        throw std::invalid_argument("CPU expert source does not match original physical NODE_LOCAL/NUMA inventory");
                    original_node=rank.node_id;members.push_back({member.world_rank,member.address});
                }
                program.emplace(NativeCPUCollectiveProgram::forBoundExpertShardDomain(
                    domain_ordinal,domain.backend,std::move(members)));
                return true;
            });
        if(!prepare)return false;
        if(!runRankSynchronizedInitializationPhase("CPU expert original domain commitment",
            {1,"cpu_expert_original_domain_commitment"},InitializationConsensusScope::ExpertOverlayContext,[&] {
                const bool included=std::any_of(program->participants().begin(),program->participants().end(),
                    [&](const auto& member){return member.world_rank==world->rank();});
                if(included)use.emplace(std::in_place_type<NativeCPUCollectiveProgram::DomainUse>,
                    program->admit(physical_memory_authority_,world->rank()));
                else use.emplace(std::in_place_type<NativeCPUCollectiveProgram::ProcessUse>,
                    program->admitExcludedParent(physical_memory_authority_,world->rank()));
                return true;
            }))return false;
        return runRankSynchronizedInitializationPhase("CPU expert native original domain",
            {2,"cpu_expert_native_original_domain"},InitializationConsensusScope::ExpertOverlayContext,[&] {
                cpu_expert_shard_ctx_=GlobalTPContext::createAdmittedCPUExpertDomain(world->communicator(),*program,std::move(*use));
                return true;
            });
    }

    bool OrchestrationRunner::
        usesNativeGpuStageMoEOverlayAuthority() const
    {
        if (moe_pipeline_prepared_plan_)
            return std::all_of(moe_pipeline_prepared_plan_->stages().begin(), moe_pipeline_prepared_plan_->stages().end(),
                [](const auto &stage) {
                    return MoEOverlayCapacityAdmission::selectMigrationStorage(*stage.placement,
                        MoERebalanceRuntimeMode::Dynamic) == MoEOverlayMigrationStorageKind::DeviceTransferDirectory;
                });
        const auto &plan = config_.moe_routed_expert_plan;
        return plan &&
               MoEOverlayCapacityAdmission::selectMigrationStorage(
                   *plan,
                   MoERebalanceRuntimeMode::Dynamic) ==
                   MoEOverlayMigrationStorageKind::
                       DeviceTransferDirectory;
    }

    bool OrchestrationRunner::
        initializeMoEExpertOverlayResidencyMaintenance()
    {
        shutdownMoEExpertOverlayResidencyMaintenance(
            MoEOverlayMaintenanceDrainIntent::ResetComposition);
        std::uint32_t maintenance_phase_ordinal = 0;
        const auto run_maintenance_phase =
            [this, &maintenance_phase_ordinal](
                const char *phase_name,
                const std::function<bool()> &step)
        {
            return runRankSynchronizedInitializationPhase(
                "ExpertOverlay maintenance initialization",
                RankInitializationPhaseIdentity{
                    .ordinal = maintenance_phase_ordinal++,
                    .name = phase_name,
                },
                InitializationConsensusScope::ExpertOverlayContext,
                step);
        };
        if (moe_pipeline_prepared_plan_)
        {
            if (!runner_ || moe_pipeline_stage_bindings_.size() != moe_pipeline_prepared_plan_->stages().size())
                return setError("MoE pipeline maintenance lost its complete stage ownership");
            runner_->requireMoEPipelineRuntimeOwnership(
                IInferenceRunner::MoEPipelineOwnershipValidation::OwnersBound);
            if (!run_maintenance_phase("finalizePipelinePreparedResidencyBanks", [&] {
                    const auto weights = model_ctx_->concreteWeightManager();
                    if (!weights) return setError("MoE pipeline has no prepared-engine registry");
                    for (const auto &binding : moe_pipeline_stage_bindings_)
                    {
                        std::string error;
                        if (!binding->owners().residency->bindOriginalPreparedContext(model_ctx_,
                                binding->owners().residency->projectionPreparation(),&error) ||
                            !binding->owners().residency->finalizeInitialBanksFromPreparedRegistry(weights->expertGemmRegistry(), &error))
                            return setError("MoE pipeline prepared banks are incomplete: " + error);
                        recordInitialMoEOverlayBankSelections(*binding->owners().residency,
                            binding->plan()->owner_order, binding->metadata().main_inference_layer_count);
                    }
                    return true;
                }))
                return false;
            if (!run_maintenance_phase("preparePipelineCPUHostMaintenance", [&] {
                    for (const auto &binding : moe_pipeline_stage_bindings_)
                        if (const auto &owner = binding->owners().host_maintenance; owner)
                            owner->prepare();
                    return true;
                })) return false;
            // Nested TP-in-PP construction prepares weights and persistent
            // state, but deliberately leaves the graph family unmaterialized.
            // Finalized expert banks must precede capture; readiness can only
            // be checked after that capture has built every stage controller.
            if (!run_maintenance_phase("materializePipelineServingGraphs",
                    [&] { return materializeCurrentServingGraphFamilyVariantWithoutLaunch(); }))
                return false;
            runner_->requireMoEPipelineRuntimeOwnership(
                IInferenceRunner::MoEPipelineOwnershipValidation::ServingMaterialized);
            for (const auto &binding : moe_pipeline_stage_bindings_)
                if (const auto &proof = binding->owners().authority->noPeerResidency(); proof)
                    PerfStatsCollector::addCounter("moe_overlay_residency", "pipeline_no_peer_ownership_sealed", 1.0,
                        "model_setup", proof->device().toString(),
                        {{"first_layer", std::to_string(binding->metadata().first_model_layer)},
                         {"authority_execution", toString(proof->authorityExecution())},
                         {"maintenance_mode", moeRebalanceRuntimeModeToString(config_.moe_rebalance.mode)},
                         {"migration_payload_bytes", "0"}, {"movement_collectives", "0"}});
            PerfStatsCollector::addCounter("moe_overlay_residency", "pipeline_maintenance_composed", 1.0,
                "model_setup", {}, {{"stages", std::to_string(moe_pipeline_stage_bindings_.size())},
                    {"authority_execution", "stage_owned"}, {"host_maintenance_service",
                        std::any_of(moe_pipeline_stage_bindings_.begin(), moe_pipeline_stage_bindings_.end(),
                            [](const auto &binding) { return bool(binding->owners().host_maintenance); }) ? "true" : "false"}});
            return true;
        }
        if (!moe_expert_overlay_residency_authority_)
            return true;
        if (!run_maintenance_phase(
                "finalizeInitialPreparedResidencyBanks",
                [&]
                {
                    if (!moe_expert_overlay_participant_residency_)
                    {
                        return setError(
                            "ExpertOverlay maintenance has no process-local "
                            "participant residency registry");
                    }
                    const auto weight_manager =
                        model_ctx_ ? model_ctx_->concreteWeightManager()
                                   : nullptr;
                    if (!weight_manager)
                    {
                        return setError(
                            "ExpertOverlay maintenance has no model-owned "
                            "prepared engine registry");
                    }

                    std::string error;
                    if (!moe_expert_overlay_participant_residency_->bindOriginalPreparedContext(model_ctx_,
                            moe_expert_overlay_participant_residency_->projectionPreparation(),&error) ||
                        !moe_expert_overlay_participant_residency_
                             ->finalizeInitialBanksFromPreparedRegistry(
                                 weight_manager->expertGemmRegistry(),
                                 &error))
                    {
                        return setError(
                            error.empty()
                                ? "ExpertOverlay could not finalize every "
                                  "process-local initial prepared bank"
                                : error);
                    }
                    return true;
                }))
        {
            return false;
        }

        const auto initial_snapshot =
            moe_expert_overlay_residency_authority_->snapshot();
        if (!initial_snapshot || !initial_snapshot->valid())
        {
            return setError(
                "ExpertOverlay maintenance requires a valid initial residency snapshot");
        }

        const int raw_layer_count = std::max(
            model_ctx_->totalBlockCount(),
            model_ctx_->blockCount());
        const int main_layer_count = mainLayerCountExcludingMTP(
            model_ctx_->concreteLoader(),
            model_ctx_->architecture(),
            raw_layer_count);
        recordInitialMoEOverlayBankSelections(
            *moe_expert_overlay_participant_residency_,
            initial_snapshot->placement_plan->owner_order,
            main_layer_count);

        std::string perf_device;
        for (const auto &tier : initial_snapshot->placement_plan->routed_tiers)
        {
            if (!perf_device.empty())
                perf_device += '/';
            perf_device += tier.name;
        }
        if (perf_device.empty())
            perf_device = "expert_overlay";

        if (moe_expert_overlay_residency_authority_->maintenanceMode() !=
            config_.moe_rebalance.mode)
        {
            return setError(
                "ExpertOverlay authority maintenance mode disagrees with the frozen runtime configuration");
        }

        if (const auto &proof = moe_expert_overlay_residency_authority_->noPeerResidency(); proof)
        {
            // The identity is frozen against every resident payload, not an
            // unmaterialized capability. No movement service or fake controller
            // is constructed for a topology with no alternative owner.
            if (!run_maintenance_phase("sealNoPeerServingGraphs", [&] {
                    return bindMoEOverlayTransferProgressAndMaterializeServingGraphFamily();
                }))
                return false;
            PerfStatsCollector::addCounter("moe_overlay_residency", "no_peer_ownership_sealed", 1.0,
                "model_setup", proof->device().toString(),
                {{"authority_execution", toString(proof->authorityExecution())},
                 {"maintenance_mode", moeRebalanceRuntimeModeToString(config_.moe_rebalance.mode)},
                 {"migration_payload_bytes", "0"}, {"movement_collectives", "0"}});
            return true;
        }

        if (moe_expert_overlay_residency_authority_->stationaryTensorShardResidency() &&
            config_.moe_rebalance.mode == MoERebalanceRuntimeMode::Dynamic)
        {
            if (!run_maintenance_phase("bindStationaryTensorShardServingGraphs", [&] {
                    return bindMoEOverlayTransferProgressAndMaterializeServingGraphFamily();
                }))
                return false;
            PerfStatsCollector::addCounter("moe_overlay_residency", "maintenance_not_started", 1.0,
                "model_setup", perf_device, {{"runtime_mode", "dynamic"},
                    {"reason", "single_fully_resident_tensor_shard_cohort"}, {"movement", "false"}});
            return true;
        }
        if(initial_snapshot->jointOwnerMap() && config_.moe_rebalance.mode==MoERebalanceRuntimeMode::Off) {
            if(config_.moe_rebalance.mode!=MoERebalanceRuntimeMode::Off ||
               moe_expert_overlay_residency_authority_->migrationEnabled() ||
               !moe_expert_overlay_participant_residency_->projectionPreparation() ||
               moe_expert_overlay_participant_residency_->projectionPreparation()->jointOwnerMap().get()!=
                   initial_snapshot->jointOwnerMap().get())
                return setError("Static CPU joint startup lost its original immutable fragment authority");
            if(!run_maintenance_phase("bindStaticJointTransferAndServingGraphs",[&] {
                return bindMoEOverlayTransferProgressAndMaterializeServingGraphFamily();
            }))return false;
            PerfStatsCollector::addCounter("moe_overlay_residency","maintenance_not_started",1.0,
                "model_setup",perf_device,{{"runtime_mode","off"},{"reason","static_joint_native_source"},{"movement","false"}});
            return true;
        }
        const auto maintenance_world = moe_expert_overlay_mpi_ctx_ ? moe_expert_overlay_mpi_ctx_ : mpi_ctx_;
        const bool local_cpu_host_maintenance = maintenance_world && maintenance_world->world_size() == 1 &&
            initial_snapshot->placement_plan->authority_execution == MoEOverlayAuthorityExecutionKind::HostResident &&
            config_.moe_rebalance.mode == MoERebalanceRuntimeMode::Dynamic &&
            std::all_of(initial_snapshot->originalOwnerMap().participants().begin(), initial_snapshot->originalOwnerMap().participants().end(),
                [](const auto &participant) { return participant.device.is_cpu(); });
        std::shared_ptr<const MoEOverlayPreparedTargetServicePlan> prepared_target_service_plan;
        std::vector<MoEOverlayPreparedTargetServiceTotals> prepared_target_service_measurements;
        if (config_.moe_rebalance.mode == MoERebalanceRuntimeMode::Dynamic &&
            !local_cpu_host_maintenance &&
            !run_maintenance_phase("declarePreparedExpertService", [&]
            {
                // This setup thread has the same MPI/NUMA/OpenMP placement as
                // CPU serving. Finish private observations before maintenance
                // workers or user requests can change the initial bank.
                const auto &mtp = retainedMTPConfig();
                const auto metadata = resolveMoERoutedExpertModelMetadataForModel(*model_ctx_, mtp);
                const auto catalog = moe_expert_overlay_residency_authority_->economyLayerCatalog();
                if (!catalog) return setError("Dynamic overlay lost its prepared-service catalog");
                const auto topology = ExpertHistogramProductionTopology::forRetainedExecution(
                    metadata.num_layers, metadata.main_inference_layer_count,
                    expertHistogramServingRegime(mtp), metadata.first_model_layer);
                prepared_target_service_plan = MoEOverlayPreparedTargetServicePlan::checked(
                    initial_snapshot, catalog, topology);
                return true;
            }))
            return false;

        if (prepared_target_service_plan)
        {
            const auto* rank_runner=dynamic_cast<const RankOrchestrator*>(runner_.get());
            const auto* local=rank_runner?rank_runner->localTPContext():nullptr;
            const auto& profile=planningModelMetadata().memoryProfile();
            const auto& inventory=cluster_inventory_.ranks.at(plan_.rank);
            const auto& source_plan=*initial_snapshot->placement_plan;
            const bool gpu_cohort=std::any_of(source_plan.domains.begin(),source_plan.domains.end(),
                [](const auto &domain) {
                    return domain.routed_compute_policy==RoutedExpertComputePolicy::TensorSharded &&
                        !domain.participants.empty() && domain.participants.front().isGPU();
                });
            std::vector<PlanningGPUCohortServiceObservation> gpu_cohort_samples;
            if (gpu_cohort && !run_maintenance_phase("measurePreparedGPUCohorts", [&] {
                MoEOverlayOriginalGPUCohortContexts contexts;
                for (const auto &domain : source_plan.domains)
                {
                    if (domain.routed_compute_policy!=RoutedExpertComputePolicy::TensorSharded ||
                        domain.participants.empty() || !domain.participants.front().isGPU() ||
                        domain.owner_rank!=plan_.rank)
                        continue;
                    if (domain.name!=source_plan.continuation_domain || !local)
                        return setError("Prepared GPU cohort lost its original LocalTP continuation");
                    contexts.addOriginalDomain(domain.name,*local);
                }
                gpu_cohort_samples=MoEOverlayCPUServiceMeasurement::measurePreparedGPUTargets(
                    *moe_expert_overlay_participant_residency_,*prepared_target_service_plan,
                    contexts,physical_memory_authority_,profile,inventory);
                return true;
            })) return false;

            if (!run_maintenance_phase("measurePreparedExpertService", [&] {
                const auto geometry=resolveMoERoutedExpertModelMetadataForModel(*model_ctx_,retainedMTPConfig());
                prepared_target_service_measurements=MoEOverlayCPUServiceMeasurement::measurePreparedTargets(
                    *moe_expert_overlay_participant_residency_,*prepared_target_service_plan,
                    {.d_model=geometry.d_model,.intermediate=geometry.routed_intermediate_size},
                    cpu_expert_shard_ctx_,physical_memory_authority_,
                    &profile,&inventory,local,profile.expert_used_count,gpu_cohort_samples);
                return true;
            })) return false;
        }

        if (local_cpu_host_maintenance)
        {
            if (!run_maintenance_phase("prepareCPUHostMaintenance", [&] {
                    const auto metadata = resolveMoERoutedExpertModelMetadataForModel(*model_ctx_, retainedMTPConfig());
                    const auto storage = resolveMoEOverlayCapacityAdmissionPolicy(*initial_snapshot->placement_plan,
                        config_, 1, metadata.num_layers, metadata.num_experts);
                    const auto profile_identity = expertOverlayMigrationProfileReuseIdentity(
                        requested_routed_weight_authority_identity_, *moe_expert_overlay_residency_authority_->economyLayerCatalog());
                    auto owner = std::make_shared<MoEOverlayCPUHostMaintenance>(MoEOverlayCPUHostMaintenance::Config{
                        .metadata = metadata, .storage = storage, .rebalance = config_.moe_rebalance,
                        .topology = ExpertHistogramProductionTopology::forRetainedExecution(metadata.num_layers,
                            metadata.main_inference_layer_count, expertHistogramServingRegime(retainedMTPConfig()), metadata.first_model_layer),
                        .memory = physical_memory_authority_, .admission = physical_memory_authority_->admission(),
                        .authority = moe_expert_overlay_residency_authority_, .residency = moe_expert_overlay_participant_residency_,
                        .histogram = moe_expert_overlay_decode_histogram_,
                        .presealed_measurements = retained_expert_overlay_migration_profile_identity_ == profile_identity
                            ? retained_expert_overlay_migration_profile_ : nullptr,
                        .perf_device = perf_device});
                    owner->prepare();
                    moe_expert_overlay_economy_calibration_ = std::shared_ptr<MoEOverlayEconomyCalibrationController>(owner, owner->calibration());
                    moe_expert_overlay_migration_profile_identity_ = profile_identity;
                    moe_expert_overlay_maintenance_service_ = std::shared_ptr<MoEOverlayResidencyMaintenanceService>(owner, owner->service());
                    moe_cpu_host_maintenance_ = std::move(owner);
                    return bindMoEOverlayTransferProgressAndMaterializeServingGraphFamily();
                })) return false;
            return true;
        }

        const bool mapped_device_authority =
            initial_snapshot->placement_plan->authority_execution ==
                MoEOverlayAuthorityExecutionKind::DeviceResident &&
            moe_overlay_device_controller_topology_ &&
            moe_overlay_device_controller_topology_
                ->usesNodeLocalMappedControl();
        if (mapped_device_authority)
        {
            try
            {
                const bool dynamic =
                    config_.moe_rebalance.mode ==
                    MoERebalanceRuntimeMode::Dynamic;
                const auto overlay_world =
                    moe_expert_overlay_mpi_ctx_
                        ? moe_expert_overlay_mpi_ctx_
                        : mpi_ctx_;
                if (!overlay_world)
                {
                    throw std::logic_error(
                        "Mapped device controller lost its production MPI identity");
                }
                std::shared_ptr<MoEOverlayEconomyCertificationController>
                    device_economy_certification;

                if (dynamic)
                {
                    const int world_size = overlay_world->world_size();
                    const MTPRuntimeConfig &mtp = retainedMTPConfig();
                    const auto metadata =
                        resolveMoERoutedExpertModelMetadataForModel(
                            *model_ctx_, mtp);
                    const auto capacity_policy = resolveMoEOverlayCapacityAdmissionPolicy(
                        *initial_snapshot->placement_plan,
                        config_,
                        world_size,
                        metadata.num_layers,
                        metadata.num_experts);
                    if (!capacity_policy.usesPhysicalResidencyFabric())
                    {
                        throw std::logic_error(
                            "Mapped Dynamic controller disagrees with setup-time physical capacity admission");
                    }
                    auto layer_weight_manifest =
                        buildMoEOverlayLayerWeightManifestFromLoader(model_ctx_->concreteLoader(),
                            metadata.num_layers,
                            metadata.num_experts);
                    const auto calibration_layer_catalog =
                        moe_expert_overlay_residency_authority_
                            ->economyLayerCatalog();
                    if (!calibration_layer_catalog ||
                        calibration_layer_catalog->layerCount() !=
                            static_cast<std::size_t>(
                                metadata.num_layers))
                    {
                        throw std::logic_error(
                            "Mapped Dynamic controller lost its canonical economy layer catalog");
                    }
                    const auto production_topology =
                        ExpertHistogramProductionTopology::
                            forRetainedExecution(
                                metadata.num_layers,
                                metadata.main_inference_layer_count,
                                expertHistogramServingRegime(mtp));
                    const auto economy_sources =
                        production_topology.economyActiveSources();
                    std::shared_ptr<IMoEOverlayEconomyEvidenceExchange>
                        economy_evidence_exchange;

                    /*
                     * Control words use the node-local mapped channel, while
                     * cross-rank weight bytes retain private asynchronous MPI
                     * lanes.  Every rank duplicates these lanes in one setup
                     * order before fallible local slot materialization.
                     */
                    if (world_size > 1)
                    {
                        const auto lane_budget=remoteMoEOverlayProjectionLaneBudget(
                        configuredMoEOverlayLaneDemand(*initial_snapshot->placement_plan,layer_weight_manifest,
                            metadata.num_experts,world_size,capacity_policy.maximum_concurrent_cycles,capacity_policy.maximum_cycles_per_layer));
                        moe_expert_overlay_remote_projection_transport_ =
                            std::make_shared<
                                MoEOverlayMPIRemoteProjectionTransport>(
                                MoEOverlayMPIRemoteProjectionTransport::Config{
                                    .mpi_context = overlay_world,
                                    .lane_budget = lane_budget,
                                    .staging_capacity_bytes =
                                        capacity_policy
                                            .staging_capacity_bytes,
                                    .perf_device = perf_device,
                                });
                        moe_expert_overlay_residency_consensus_ =
                            std::make_shared<
                                MoEOverlayMPIResidencyConsensus>(
                                MoEOverlayMPIResidencyConsensus::Config{
                                    .mpi_context = overlay_world,
                                    .perf_device = perf_device,
                                });
                        economy_evidence_exchange = std::make_shared<
                            MoEOverlayMPIEconomyEvidenceExchange>(
                            MoEOverlayMPIEconomyEvidenceExchange::Config{
                                .mpi_context = overlay_world,
                                .owner_map = initial_snapshot->originalOwnerMap(),
                                .production_topology = production_topology,
                                .perf_device = perf_device,
                            });
                    }

                    if (!run_maintenance_phase(
                            "materializeMappedDevicePhysicalFabric",
                            [&]
                            {
                                try
                                {
                                    moe_expert_overlay_physical_residency_fabric_ =
                                        MoEOverlayPhysicalResidencyFabric::create({
                                            .memory_authority =
                                                physical_memory_authority_,
                                            .registry =
                                                moe_expert_overlay_participant_residency_,
                                            .initial_snapshot = initial_snapshot,
                                            .physical_gpu_lane_demand = moe_expert_overlay_memory_admission_
                                                ? moe_expert_overlay_memory_admission_->physical_gpu_lane_demand : nullptr,
                                            .inventory_authority =
                                                MoEOverlayPhysicalInventoryAuthority::
                                                    DeviceSlotLedger,
                                            .layer_weight_manifest =
                                                std::move(layer_weight_manifest),
                                            .remote_projection_transport =
                                                moe_expert_overlay_remote_projection_transport_,
                                            .shadow_slots_per_endpoint_layer =
                                                capacity_policy
                                                    .shadow_slots_per_endpoint_layer,
                                            .staging_capacity_bytes =
                                                capacity_policy
                                                    .staging_capacity_bytes,
                                            .maximum_concurrent_cycles =
                                                capacity_policy
                                                    .maximum_concurrent_cycles,
                                            .maximum_execution_streams =
                                                capacity_policy
                                                    .maximum_execution_streams,
                                    .maximum_cycles_per_layer = capacity_policy
                                                    .maximum_cycles_per_layer,
                                            .collect_economy_measurements = true,
                                            .economy_measurement_scope =
                                                MoEOverlayMigrationMeasurementScope::
                                                    EconomyCalibrationOnly,
                                            .perf_device = perf_device,
                                        });
                                    return true;
                                }
                                catch (const std::exception &error)
                                {
                                    return setError(
                                        std::string(
                                            "Mapped Dynamic physical fabric materialization failed: ") +
                                        error.what());
                                }
                                catch (...)
                                {
                                    return setError(
                                        "Mapped Dynamic physical fabric materialization raised a non-standard exception");
                                }
                            }))
                    {
                        return false;
                    }

                    moe_expert_overlay_wave_factory_ =
                        std::make_unique<
                            MoEOverlayParticipantPreparedWaveFactory>(
                            MoEOverlayParticipantPreparedWaveFactory::Config{
                                .registry =
                                    moe_expert_overlay_participant_residency_,
                                .transfer_provider =
                                    moe_expert_overlay_physical_residency_fabric_,
                                .perf_device = perf_device,
                            });
                    const auto maximum_migrations_per_wave =
                        MoEOverlayRemoteProjectionLaneBudget::fromUniformCycles(std::max<std::size_t>( 1u, initial_snapshot->originalOwnerMap() .participants().size()), capacity_policy.maximum_concurrent_cycles, 1u)
                            .projectionOperationCount();
                    if (!maximum_migrations_per_wave)
                    {
                        throw std::overflow_error(
                            "Mapped Dynamic economy journal capacity overflowed");
                    }
                    auto measurement_journal = std::make_shared<
                        MoEOverlayMigrationMeasurementJournal>(
                        MoEOverlayMigrationMeasurementJournal::Config{
                            .maximum_migrations_per_wave =
                                *maximum_migrations_per_wave,
                        });
                    std::shared_ptr<IMoEOverlayResidencyTransport>
                        calibration_transport;
                    moe_expert_overlay_migration_transport_ =
                        std::make_shared<MoEOverlayTierMigrationTransport>(
                            MoEOverlayTierMigrationTransport::Config{
                                .factory =
                                    moe_expert_overlay_wave_factory_.get(),
                                .transfer_progress_authority =
                                    moe_expert_overlay_physical_residency_fabric_,
                                .projections_per_expert =
                                    kMoEOverlayExpertProjectionCount,
                                .measurement_sink = measurement_journal,
                                .measurement_scope =
                                    MoEOverlayMigrationMeasurementScope::
                                        EconomyCalibrationOnly,
                                .require_complete_local_measurements =
                                    world_size == 1,
                                .perf_device = perf_device,
                            });
                    calibration_transport =
                        moe_expert_overlay_migration_transport_;
                    if (world_size > 1)
                    {
                        moe_expert_overlay_distributed_migration_transport_ =
                            std::make_shared<
                                MoEOverlayDistributedResidencyTransport>(
                                MoEOverlayDistributedResidencyTransport::Config{
                                    .local_transport =
                                        moe_expert_overlay_migration_transport_
                                            .get(),
                                    .consensus =
                                        moe_expert_overlay_residency_consensus_,
                                    .perf_device = perf_device,
                                });
                        calibration_transport =
                            moe_expert_overlay_distributed_migration_transport_;
                    }

                    auto calibration_planner = std::make_shared<
                        MoEOverlayEconomyCalibrationPlanner>(
                        MoEOverlayEconomyCalibrationPlanner::Config{
                            .live_snapshot = initial_snapshot,
                            .complete_expert_bytes_per_layer =
                                calibration_layer_catalog
                                    ->completeExpertBytesPerLayer(),
                            .calibration_layers =
                                calibration_layer_catalog
                                    ->representativeLayers(),
                        });
                    auto calibration_ledger = std::make_shared<
                        MoEOverlayMigrationMeasurementLedger>(
                        MoEOverlayMigrationMeasurementLedger::Config{
                            .required_coordinates =
                                calibration_planner->requiredCoordinates(),
                            .warmup_samples_per_coordinate = 0u,
                            .measured_samples_per_coordinate =
                                MoEOverlayEconomyProfileComposer::
                                    kMinimumMigrationSamples,
                                .required_sources = economy_sources,
                            .measurement_identity =
                                calibration_layer_catalog->identity(),
                        });
                    const std::string calibration_reuse_identity =
                        expertOverlayMigrationProfileReuseIdentity(
                            requested_routed_weight_authority_identity_,
                            *calibration_layer_catalog);
                    const auto reusable_measurements =
                        retained_expert_overlay_migration_profile_ &&
                                retained_expert_overlay_migration_profile_identity_ ==
                                    calibration_reuse_identity
                            ? retained_expert_overlay_migration_profile_
                            : nullptr;
                    auto calibration = std::make_shared<
                        MoEOverlayEconomyCalibrationController>(
                        MoEOverlayEconomyCalibrationController::Config{
                            .planner = std::move(calibration_planner),
                            .ledger = std::move(calibration_ledger),
                            .journal = std::move(measurement_journal),
                            .transport =
                                std::move(calibration_transport),
                            .evidence_exchange =
                                economy_evidence_exchange,
                            .presealed_measurements =
                                reusable_measurements,
                            .perf_device = perf_device,
                        });
                    moe_expert_overlay_economy_calibration_ = calibration;
                    moe_expert_overlay_migration_profile_identity_ =
                        calibration_reuse_identity;
                    device_economy_certification = std::make_shared<
                        MoEOverlayEconomyCertificationController>(
                            MoEOverlayEconomyCertificationController::Config{
                            .calibration = calibration,
                            .registry =
                                moe_expert_overlay_participant_residency_,
                            .layer_catalog =
                                std::move(calibration_layer_catalog),
                            .authority =
                                moe_expert_overlay_residency_authority_,
                            .target =
                                MoEOverlayEconomyCertificationTarget::
                                    DetachedDeviceAuthority,
                            .model_metadata = metadata,
                            .economy_policy =
                                MoEOverlayMigrationEconomyPolicy{
                                    .payoff_horizon_tokens =
                                        config_.moe_rebalance
                                            .migration_payoff_horizon_tokens,
                                },
                            .production_topology = production_topology,
                            .prepared_target_service_plan = prepared_target_service_plan,
                            .prepared_target_service_measurements = prepared_target_service_measurements,
                            .prepared_service_world_rank = physical_memory_authority_->worldRank(),
                            .evidence_exchange =
                                std::move(economy_evidence_exchange),
                            .perf_device = perf_device,
                        });
                }

                /*
                 * A serving graph may capture the transfer epoch only after
                 * the fabric has allocated its final slots and exact streams.
                 * Seal both continuation and follower families before the
                 * controller service can publish a maintenance command.
                 */
                if (!run_maintenance_phase(
                        "bindTransferProgressAndServingGraphs",
                        [&]
                        {
                            return bindMoEOverlayTransferProgressAndMaterializeServingGraphFamily();
                        }))
                    return false;

                moe_overlay_device_controller_graph_service_ =
                    std::make_unique<
                        MoEOverlayDeviceControllerGraphService>(
                        MoEOverlayDeviceControllerGraphService::Config{
                            .mpi_ctx =
                                moe_expert_overlay_mpi_ctx_
                                    ? moe_expert_overlay_mpi_ctx_
                                    : mpi_ctx_,
                            .topology =
                                moe_overlay_device_controller_topology_,
                            .fabric =
                                moe_overlay_device_controller_fabric_,
                            .physical_memory_authority = physical_memory_authority_,
                            .runtime_bindings = runner_
                                ->moeOverlayDeviceControllerRuntimeBindings(),
                            .execution_mode = dynamic
                                ? MoEOverlayDeviceControllerGraphService::
                                      ExecutionMode::Dynamic
                                : MoEOverlayDeviceControllerGraphService::
                                      ExecutionMode::Static,
                            .maintenance_window_tokens =
                                static_cast<std::uint64_t>(
                                    config_.moe_rebalance.window_size),
                            .maintenance_max_window_tokens =
                                static_cast<std::uint64_t>(
                                    config_.moe_rebalance.max_window_size),
                            .maintenance_window_growth_factor =
                                static_cast<double>(
                                    config_.moe_rebalance
                                        .window_growth_factor),
                            .physical_fabric =
                                moe_expert_overlay_physical_residency_fabric_,
                            .economy_certification =
                                std::move(device_economy_certification),
                            .perf_device = perf_device,
                        });
                if (!dynamic)
                {
                    std::string certification_error;
                    if (!moe_overlay_device_controller_graph_service_
                             ->certifyStaticNoMovement(
                                 &certification_error))
                    {
                        return setError(
                            certification_error.empty()
                                ? "Captured multi-group device controller failed its Static no-movement certification"
                                : certification_error);
                    }
                }
            }
            catch (const std::exception &error)
            {
                return setError(
                    std::string(
                        "Failed to materialize captured multi-group ExpertOverlay controller graphs: ") +
                    error.what());
            }

            PerfStatsCollector::addCounter(
                "moe_overlay_residency",
                "production_maintenance_composed",
                1.0,
                "model_setup",
                perf_device,
                {{"authority_execution", "device_resident"},
                 {"controller_transport", "node_local_mapped"},
                 {"background", "dedicated_device_stream"},
                 {"host_maintenance_service", "false"},
                 {"durable_writer_count", "1"},
                 {"movement",
                  config_.moe_rebalance.mode ==
                          MoERebalanceRuntimeMode::Dynamic
                      ? "true"
                      : "false"},
                 {"physical_fabric",
                  moe_expert_overlay_physical_residency_fabric_
                      ? "pre_materialized"
                      : "none"}});
            return true;
        }

        if (!moe_expert_overlay_residency_authority_
                 ->migrationEnabled())
        {
            /*
             * Off and Observe have no physical movement authority. Execute
             * the typed immobility check once so PerfStats proves zero
             * migrations without allocating a fabric or polling thread.
             */
            StaticMoEOverlayTransportGuard guard;
            const auto transaction =
                moe_expert_overlay_residency_authority_
                    ->proposeFromHistogram();
            const auto result =
                moe_expert_overlay_residency_authority_->beginApply(
                    transaction, guard);
            if (result.status !=
                MoEOverlayResidencyApplyStatus::StaticNoMovement)
            {
                return setError(
                    result.error.empty()
                        ? "Movement-disabled ExpertOverlay maintenance did not prove zero movement"
                        : result.error);
            }
            PerfStatsCollector::addCounter(
                "moe_overlay_residency",
                "maintenance_not_started",
                1.0,
                "model_setup",
                perf_device,
                {{"runtime_mode",
                  moeRebalanceRuntimeModeToString(
                      config_.moe_rebalance.mode)},
                 {"reason", "explicit_runtime_policy"},
                 {"movement", "false"}});
            if (!run_maintenance_phase(
                    "bindTransferProgressAndServingGraphs",
                    [&]
                    {
                        return bindMoEOverlayTransferProgressAndMaterializeServingGraphFamily();
                    }))
                return false;
            return true;
        }

        if (usesNativeGpuStageMoEOverlayAuthority())
        {
            if (!runner_)
            {
                return setError(
                    "Native GPU-stage device-resident ExpertOverlay authority has no constructed inference runner");
            }
            const auto runner_execution =
                runner_->moeOverlayAuthorityExecution();
            if (runner_execution !=
                MoEOverlayAuthorityExecutionKind::
                    DeviceResident)
            {
                return setError(
                    "ExpertOverlay graph family disagrees with the frozen single-domain native device-resident authority selection");
            }
            if (!runner_
                     ->deviceResidentMoEOverlayMaintenanceReady())
            {
                return setError(
                    "Native GPU-stage device-resident ExpertOverlay did not materialize its complete captured maintenance family");
            }

            if (!run_maintenance_phase(
                    "bindTransferProgressAndServingGraphs",
                    [&]
                    {
                        return bindMoEOverlayTransferProgressAndMaterializeServingGraphFamily();
                    }))
                return false;

            /*
             * The Qwen graph family has already materialized the captured
             * all-layer transaction and its NCCL/RCCL transfer stream. Do not
             * construct a host fabric or maintenance thread here: that would
             * create a second planner and a second durable epoch writer.
             */
            PerfStatsCollector::addCounter(
                "moe_overlay_residency",
                "production_maintenance_composed",
                1.0,
                "model_setup",
                perf_device,
                {{"authority_execution", "device_resident"},
                 {"background", "device_stream"},
                 {"host_maintenance_service", "false"},
                 {"durable_writer_count", "1"}});
            return true;
        }

        const auto local_ids =
            moe_expert_overlay_participant_residency_
                ->localParticipantIds();

        try
        {
            const auto overlay_world =
                moe_expert_overlay_mpi_ctx_
                    ? moe_expert_overlay_mpi_ctx_
                    : mpi_ctx_;
            const int world_rank = overlay_world ? overlay_world->rank() : 0;
            const int world_size =
                overlay_world ? overlay_world->world_size() : 1;
            const bool distributed = world_size > 1;
            const MTPRuntimeConfig &mtp = retainedMTPConfig();
            const auto metadata =
                resolveMoERoutedExpertModelMetadataForModel(
                    *model_ctx_,
                    mtp);
            const auto capacity_policy = resolveMoEOverlayCapacityAdmissionPolicy(
                *initial_snapshot->placement_plan,
                config_,
                world_size,
                metadata.num_layers,
                metadata.num_experts);
            if (!capacity_policy.usesPhysicalResidencyFabric())
            {
                throw std::logic_error(
                    "Dynamic ExpertOverlay maintenance disagrees with its setup-time capacity policy");
            }
            const auto production_topology =
                ExpertHistogramProductionTopology::forRetainedExecution(
                    metadata.num_layers,
                    metadata.main_inference_layer_count,
                    expertHistogramServingRegime(mtp), metadata.first_model_layer);
            const auto economy_sources =
                production_topology.economyActiveSources();
            const auto layer_weight_manifest =
                buildMoEOverlayLayerWeightManifestFromLoader(model_ctx_->concreteLoader(),
                    metadata.num_layers,
                    metadata.num_experts, metadata.first_model_layer);
            const auto calibration_layer_catalog =
                moe_expert_overlay_residency_authority_
                    ->economyLayerCatalog();
            if (!calibration_layer_catalog ||
                calibration_layer_catalog->layerCount() !=
                    static_cast<std::size_t>(metadata.num_layers))
            {
                throw std::logic_error(
                    "Dynamic ExpertOverlay maintenance lost its canonical economy layer catalog");
            }

            const auto execution_plan =
                resolveMoEExpertOverlayExecutionPlan(
                    initial_snapshot->placement_plan,
                    MoEExpertOverlayExecutionPlanResolverOptions{
                        .current_world_rank = world_rank,
                        .world_size = world_size,
                    });
            std::shared_ptr<IMoEOverlayEconomyEvidenceExchange>
                economy_evidence_exchange;

            if (distributed)
            {
                if (!overlay_world)
                {
                    throw std::logic_error(
                        "Distributed ExpertOverlay maintenance lost its overlay MPI context");
                }
                if (!moe_expert_overlay_decode_histogram_)
                {
                    throw std::logic_error(
                        "Distributed ExpertOverlay maintenance lost its admitted routing histogram");
                }

                /*
                 * One simple closed cycle visits each physical participant at
                 * most once, but policy may admit several cycles from
                 * different layers in one wave. Size the remote projection
                 * pool from all three factors so raising the public cycle cap
                 * also raises its exact setup-time lane BOM. Every MPI lane
                 * and receive staging buffer is materialized now, never on
                 * maintenance or inference paths.
                 */
                const auto lane_budget=remoteMoEOverlayProjectionLaneBudget(
                        configuredMoEOverlayLaneDemand(*initial_snapshot->placement_plan,layer_weight_manifest,
                            metadata.num_experts,world_size,capacity_policy.maximum_concurrent_cycles,capacity_policy.maximum_cycles_per_layer));
                /*
                 * Duplicate every private communicator in one invariant order
                 * on every rank before fallible rank-local slot construction.
                 * A local VRAM or format error can then unwind idle lanes
                 * without leaving a peer blocked in a later collective dup.
                 */
                moe_expert_overlay_residency_consensus_ =
                    std::make_shared<MoEOverlayMPIResidencyConsensus>(
                        MoEOverlayMPIResidencyConsensus::Config{
                            .mpi_context = overlay_world,
                            .perf_device = perf_device,
                        });
                moe_expert_overlay_remote_projection_transport_ =
                    std::make_shared<
                        MoEOverlayMPIRemoteProjectionTransport>(
                        MoEOverlayMPIRemoteProjectionTransport::Config{
                            .mpi_context = overlay_world,
                            .lane_budget = lane_budget,
                            .staging_capacity_bytes =
                                capacity_policy.staging_capacity_bytes,
                            .perf_device = perf_device,
                        });
                moe_expert_overlay_proposal_publisher_ =
                    std::make_shared<
                        MoEOverlayMPIResidencyProposalPublisher>(
                        MoEOverlayMPIResidencyProposalPublisher::Config{
                            .mpi_context = overlay_world,
                            .coordinator_world_rank =
                                execution_plan.continuation_root_rank,
                            .num_layers = metadata.num_layers,
                            .num_experts = metadata.num_experts,
                            .perf_device = perf_device,
                            // The routing owner and its mailbox share one
                            // admitted geometry, never independent row caps.
                            .transaction_demand =
                                moe_expert_overlay_decode_histogram_
                                    ->config().transaction_demand,
                            .first_model_layer = metadata.first_model_layer,
                        });
                economy_evidence_exchange = std::make_shared<
                    MoEOverlayMPIEconomyEvidenceExchange>(
                    MoEOverlayMPIEconomyEvidenceExchange::Config{
                        .mpi_context = overlay_world,
                        .owner_map = initial_snapshot->originalOwnerMap(),
                        .production_topology = production_topology,
                        .perf_device = perf_device,
                    });
            }

            std::shared_ptr<IMoEOverlayResidencyTransport>
                maintenance_transport;
            std::shared_ptr<MoEOverlayMigrationMeasurementJournal>
                local_measurement_journal;
            std::shared_ptr<MoEOverlayEconomyCertificationController>
                local_economy_certification;
            if (!run_maintenance_phase(
                    "materializeHostAuthorityPhysicalFabric",
                    [&]
                    {
                        try
                        {
                            moe_expert_overlay_physical_residency_fabric_ =
                                MoEOverlayPhysicalResidencyFabric::create({
                                    .memory_authority =
                                        physical_memory_authority_,
                                    .registry =
                                        moe_expert_overlay_participant_residency_,
                                    .initial_snapshot = initial_snapshot,
                                    .physical_gpu_lane_demand = moe_expert_overlay_memory_admission_
                                        ? moe_expert_overlay_memory_admission_->physical_gpu_lane_demand : nullptr,
                                    .inventory_authority =
                                        MoEOverlayPhysicalInventoryAuthority::
                                            ParticipantRegistry,
                                    // Calibration subsequently authenticates the same
                                    // immutable complete source descriptors.
                                    .layer_weight_manifest =
                                        layer_weight_manifest,
                                    .remote_projection_transport =
                                        moe_expert_overlay_remote_projection_transport_,
                                    .shadow_slots_per_endpoint_layer =
                                        capacity_policy
                                            .shadow_slots_per_endpoint_layer,
                                    .staging_capacity_bytes =
                                        capacity_policy.staging_capacity_bytes,
                                    .maximum_concurrent_cycles =
                                        capacity_policy
                                            .maximum_concurrent_cycles,
                                    .maximum_execution_streams =
                                        capacity_policy
                                            .maximum_execution_streams,
                                    .maximum_cycles_per_layer = capacity_policy
                                            .maximum_cycles_per_layer,
                                    .collect_economy_measurements = true,
                                    .economy_measurement_scope =
                                        MoEOverlayMigrationMeasurementScope::
                                            EconomyCalibrationOnly,
                                    .perf_device = perf_device,
                                });
                            return true;
                        }
                        catch (const std::exception &error)
                        {
                            return setError(
                                std::string(
                                    "Host-authority physical fabric materialization failed: ") +
                                error.what());
                        }
                        catch (...)
                        {
                            return setError(
                                "Host-authority physical fabric materialization raised a non-standard exception");
                        }
                    }))
            {
                return false;
            }

            PerfStatsCollector::addCounter(
                "moe_overlay_residency",
                "production_materialization_consensus",
                1.0,
                "model_setup",
                perf_device,
                {{"world_rank", std::to_string(world_rank)},
                 {"world_size", std::to_string(world_size)},
                 {"consensus", "typed_rank_phase"},
                 {"blocking", "true"},
                 {"inference_path", "false"}});

            /*
             * Physical slots, transfer streams, and their device epochs now
             * have stable lifetime. Capture them into every serving graph
             * before calibration or maintenance can enqueue migration work.
             */
            if (!run_maintenance_phase(
                    "bindTransferProgressAndServingGraphs",
                    [&]
                    {
                        return bindMoEOverlayTransferProgressAndMaterializeServingGraphFamily();
                    }))
                return false;

            /*
             * Runtime-table addresses and inference boundaries become final
             * only after the serving graph family is materialized.  Prove the
             * complete process-local GPU set now, retain each mapped
             * publication page/stream/event, and only then expose a migration
             * factory.  A wave therefore cannot exist without its device-bank
             * authority.
             */
            std::vector<MoEOverlayDeviceControllerRuntimeBinding>
                host_authority_runtime_bindings;
            std::string publication_materialization_error;
            try
            {
                if (!runner_)
                {
                    throw std::logic_error(
                        "Host-authority Dynamic ExpertOverlay has no serving runner");
                }
                host_authority_runtime_bindings =
                    runner_->moeOverlayDeviceControllerRuntimeBindings();
                std::sort(
                    host_authority_runtime_bindings.begin(),
                    host_authority_runtime_bindings.end(),
                    [](const auto &left, const auto &right)
                    {
                        return left.overlay_participant_id <
                               right.overlay_participant_id;
                    });

                std::vector<int> expected_local_gpu_participants;
                for (const auto &participant :
                     initial_snapshot->originalOwnerMap().participants())
                {
                    if (participant.device.is_gpu() &&
                        (!participant.world_rank_known ||
                         participant.world_rank == world_rank))
                    {
                        expected_local_gpu_participants.push_back(
                            participant.participant_id);
                    }
                }
                std::sort(
                    expected_local_gpu_participants.begin(),
                    expected_local_gpu_participants.end());
                std::vector<int> observed_local_gpu_participants;
                observed_local_gpu_participants.reserve(
                    host_authority_runtime_bindings.size());
                for (const auto &binding :
                     host_authority_runtime_bindings)
                {
                    if (!binding.hostPublicationValid())
                    {
                        throw std::logic_error(
                            "Host-authority Dynamic ExpertOverlay produced an incomplete GPU bank-publication binding");
                    }
                    observed_local_gpu_participants.push_back(
                        binding.overlay_participant_id);
                }
                if (observed_local_gpu_participants !=
                    expected_local_gpu_participants)
                {
                    throw std::logic_error(
                        "Host-authority Dynamic ExpertOverlay omitted a local GPU bank publisher");
                }

                const auto participant_list = [](const std::vector<int> &ids)
                {
                    std::ostringstream stream;
                    for (std::size_t index = 0u; index < ids.size(); ++index)
                    {
                        if (index != 0u)
                            stream << ',';
                        stream << ids[index];
                    }
                    return ids.empty() ? std::string{"none"} :
                                         stream.str();
                };
                LOG_INFO(
                    "[ExpertOverlay] Host-authority GPU bank publication topology"
                    << " world_rank=" << world_rank
                    << " expected_participants="
                    << participant_list(expected_local_gpu_participants)
                    << " runtime_bindings="
                    << participant_list(observed_local_gpu_participants));

                if (!host_authority_runtime_bindings.empty())
                {
                    moe_expert_overlay_device_bank_publisher_ =
                        std::make_shared<
                            MoEOverlayHostAuthorityDeviceBankPublisher>(
                            MoEOverlayHostAuthorityDeviceBankPublisher::Config{
                                .runtime_bindings =
                                    host_authority_runtime_bindings,
                                .registry =
                                    moe_expert_overlay_participant_residency_,
                                .perf_device = perf_device,
                            });
                }

                moe_expert_overlay_wave_factory_ =
                    std::make_unique<
                        MoEOverlayParticipantPreparedWaveFactory>(
                        MoEOverlayParticipantPreparedWaveFactory::Config{
                            .registry =
                                moe_expert_overlay_participant_residency_,
                            .transfer_provider =
                                moe_expert_overlay_physical_residency_fabric_,
                            .device_bank_publisher =
                                moe_expert_overlay_device_bank_publisher_,
                            .perf_device = perf_device,
                        });
                const auto maximum_migrations_per_wave =
                    MoEOverlayRemoteProjectionLaneBudget::fromUniformCycles(std::max<std::size_t>( 1, initial_snapshot->originalOwnerMap() .participants().size()), capacity_policy.maximum_concurrent_cycles, 1)
                        .projectionOperationCount();
                if (!maximum_migrations_per_wave)
                {
                    throw std::overflow_error(
                        "ExpertOverlay migration measurement journal BOM overflowed");
                }
                local_measurement_journal = std::make_shared<
                    MoEOverlayMigrationMeasurementJournal>(
                    MoEOverlayMigrationMeasurementJournal::Config{
                        .maximum_migrations_per_wave =
                            *maximum_migrations_per_wave,
                    });
                maintenance_transport =
                    moe_expert_overlay_migration_transport_ =
                        std::make_shared<MoEOverlayTierMigrationTransport>(
                            MoEOverlayTierMigrationTransport::Config{
                                .factory =
                                    moe_expert_overlay_wave_factory_.get(),
                                .transfer_progress_authority =
                                    moe_expert_overlay_physical_residency_fabric_,
                                .projections_per_expert =
                                    kMoEOverlayExpertProjectionCount,
                                .measurement_sink =
                                    local_measurement_journal,
                                .measurement_scope =
                                    MoEOverlayMigrationMeasurementScope::
                                        EconomyCalibrationOnly,
                                .require_complete_local_measurements =
                                    !distributed,
                                .perf_device = perf_device,
                            });

                if (distributed)
                {
                    moe_expert_overlay_distributed_migration_transport_ =
                        std::make_shared<
                            MoEOverlayDistributedResidencyTransport>(
                            MoEOverlayDistributedResidencyTransport::Config{
                                .local_transport =
                                    moe_expert_overlay_migration_transport_.get(),
                                .consensus =
                                    moe_expert_overlay_residency_consensus_,
                                .perf_device = perf_device,
                            });
                    maintenance_transport =
                        moe_expert_overlay_distributed_migration_transport_;
                }
            }
            catch (const std::exception &error)
            {
                publication_materialization_error = error.what();
            }
            catch (...)
            {
                publication_materialization_error =
                    "non-standard exception during host-authority GPU publication materialization";
            }

            if (!run_maintenance_phase(
                    "materializeHostAuthorityGpuPublication",
                    [&]
                    {
                        if (publication_materialization_error.empty())
                            return true;
                        return setError(
                            std::string(
                                "Host-authority GPU publication materialization failed: ") +
                            publication_materialization_error);
                    }))
            {
                return false;
            }

            {
                auto calibration_planner = std::make_shared<
                    MoEOverlayEconomyCalibrationPlanner>(
                    MoEOverlayEconomyCalibrationPlanner::Config{
                        .live_snapshot = initial_snapshot,
                        .complete_expert_bytes_per_layer =
                            calibration_layer_catalog
                                ->completeExpertBytesPerLayer(),
                        .calibration_layers =
                            calibration_layer_catalog
                                ->representativeLayers(),
                    });
                auto calibration_ledger = std::make_shared<
                    MoEOverlayMigrationMeasurementLedger>(
                    MoEOverlayMigrationMeasurementLedger::Config{
                        .required_target_measurements = initial_snapshot->jointOwnerMap()
                            ? calibration_planner->requiredTargetMeasurements(layer_weight_manifest)
                            : std::vector<MoEOverlayRequiredTargetMigrationMeasurement>{},
                        .required_coordinates = initial_snapshot->jointOwnerMap()
                            ? std::vector<MoEOverlayMigrationMeasurementCoordinate>{}
                            : calibration_planner->requiredCoordinates(),
                        .warmup_samples_per_coordinate = 0,
                        .measured_samples_per_coordinate =
                            MoEOverlayEconomyProfileComposer::
                                kMinimumMigrationSamples,
                            .required_sources = economy_sources,
                        .measurement_identity =
                            calibration_layer_catalog->identity(),
                    });
                const std::string calibration_reuse_identity =
                    expertOverlayMigrationProfileReuseIdentity(
                        requested_routed_weight_authority_identity_,
                        *calibration_layer_catalog);
                const auto reusable_measurements =
                    retained_expert_overlay_migration_profile_ &&
                            retained_expert_overlay_migration_profile_identity_ ==
                                calibration_reuse_identity
                        ? retained_expert_overlay_migration_profile_
                        : nullptr;
                auto calibration = std::make_shared<
                    MoEOverlayEconomyCalibrationController>(
                    MoEOverlayEconomyCalibrationController::Config{
                        .planner = std::move(calibration_planner),
                        .ledger = std::move(calibration_ledger),
                        .journal = local_measurement_journal,
                        .transport = maintenance_transport,
                        .evidence_exchange = economy_evidence_exchange,
                        .presealed_measurements = reusable_measurements,
                        .perf_device = perf_device,
                    });
                moe_expert_overlay_economy_calibration_ = calibration;
                moe_expert_overlay_migration_profile_identity_ =
                    calibration_reuse_identity;
                local_economy_certification = std::make_shared<
                    MoEOverlayEconomyCertificationController>(
                        MoEOverlayEconomyCertificationController::Config{
                        .calibration = calibration,
                        .registry =
                            moe_expert_overlay_participant_residency_,
                        .layer_catalog =
                            std::move(calibration_layer_catalog),
                        .authority =
                            moe_expert_overlay_residency_authority_,
                        .model_metadata = metadata,
                        .economy_policy =
                            MoEOverlayMigrationEconomyPolicy{
                                .payoff_horizon_tokens =
                                    config_.moe_rebalance
                                        .migration_payoff_horizon_tokens,
                            },
                        .production_topology = production_topology,
                        .prepared_target_service_plan = prepared_target_service_plan,
                            .prepared_target_service_measurements = prepared_target_service_measurements,
                            .prepared_service_world_rank = physical_memory_authority_->worldRank(),
                        .evidence_exchange =
                            economy_evidence_exchange,
                        .perf_device = perf_device,
                    });
            }

            std::shared_ptr<MoEOverlayDeviceServiceTelemetryPublisher>
                local_device_service_telemetry_publisher;
            if (local_economy_certification && !initial_snapshot->jointOwnerMap())
            {
                for (const auto &binding :
                     host_authority_runtime_bindings)
                {
                    if (!binding.serviceTelemetryValid() ||
                        !binding.inference_boundary)
                    {
                        throw std::logic_error(
                            "Host-authority Dynamic ExpertOverlay produced an incomplete GPU service telemetry binding");
                    }
                }
                if (!host_authority_runtime_bindings.empty())
                {
                    local_device_service_telemetry_publisher =
                        std::make_shared<
                            MoEOverlayDeviceServiceTelemetryPublisher>(
                            MoEOverlayDeviceServiceTelemetryPublisher::Config{
                                .runtime_bindings =
                                    host_authority_runtime_bindings,
                                .minimum_snapshot_interval =
                                    std::chrono::milliseconds(100),
                                .perf_device = perf_device,
                            });
                }
            }

            moe_expert_overlay_maintenance_service_ =
                std::make_unique<
                    MoEOverlayResidencyMaintenanceService>(
                    MoEOverlayResidencyMaintenanceService::Config{
                        .authority =
                            moe_expert_overlay_residency_authority_,
                        .transport =
                            std::move(maintenance_transport),
                        .economy_certification =
                            std::move(local_economy_certification),
                        .device_service_telemetry_publisher =
                            std::move(
                                local_device_service_telemetry_publisher),
                        .proposal_publisher =
                            moe_expert_overlay_proposal_publisher_,
                        .distributed_context =
                            distributed ? overlay_world : nullptr,
                        .idle_poll_interval =
                            std::chrono::milliseconds(2),
                        .perf_device = perf_device,
                    });

            PerfStatsCollector::addCounter(
                "moe_overlay_residency",
                "production_maintenance_composed",
                1.0,
                "model_setup",
                perf_device,
                {{"shadow_slots_per_endpoint_layer",
                  std::to_string(
                      capacity_policy.shadow_slots_per_endpoint_layer)},
                 {"migration_transfer_slots",
                  std::to_string(
                      config_.moe_rebalance
                          .migration_transfer_slots)},
                 {"migration_cycles_per_wave",
                  std::to_string(
                      config_.moe_rebalance
                          .resolvedMigrationCyclesPerWave())},
                 {"max_cycles_per_layer",
                  std::to_string(
                      capacity_policy.maximum_cycles_per_layer)},
                 {"staging_bytes",
                  std::to_string(
                      capacity_policy.staging_capacity_bytes)},
                 {"participants",
                  std::to_string(local_ids.size())},
                 {"global_participants",
                  std::to_string(
                      initial_snapshot->originalOwnerMap().participants().size())},
                 {"world_rank", std::to_string(world_rank)},
                 {"world_size", std::to_string(world_size)},
                 {"distributed", distributed ? "true" : "false"},
                 {"histogram_coordinator_rank",
                  std::to_string(execution_plan.continuation_root_rank)},
                 {"background", "true"}});
        }
        catch (const std::exception &error)
        {
            shutdownMoEExpertOverlayResidencyMaintenance(
                MoEOverlayMaintenanceDrainIntent::ResetComposition);
            return setError(
                std::string(
                    "Failed to compose production ExpertOverlay residency maintenance: ") +
                error.what());
        }
        return true;
    }

    bool OrchestrationRunner::
        startMoEExpertOverlayResidencyMaintenance()
    {
        try
        {
            for (const auto &binding : moe_pipeline_stage_bindings_)
                if (const auto &owner = binding->owners().host_maintenance; owner) owner->start();
        }
        catch (const std::exception &error) { return setError("CPU pipeline maintenance activation failed: " + std::string(error.what())); }
        if (moe_overlay_device_controller_graph_service_ &&
            moe_expert_overlay_maintenance_service_)
        {
            return setError(
                "ExpertOverlay composed both device- and host-resident maintenance authorities");
        }
        if (moe_overlay_device_controller_graph_service_)
        {
            if (moe_overlay_device_controller_graph_service_->state() !=
                MoEOverlayDeviceControllerActivationState::Prepared)
            {
                return setError(
                    "ExpertOverlay device maintenance reached activation outside Prepared");
            }
            try
            {
                /*
                 * This phase follows diagnostic/lean serving-family capture
                 * and initial MTP transaction setup under rank consensus.
                 * Every permitted later native capture publishes a typed
                 * activity edge before it begins, so the submitter observes
                 * Pending instead of touching that exact CUDA/HIP stream.
                 */
                moe_overlay_device_controller_graph_service_->start();
            }
            catch (const std::exception &error)
            {
                return setError(
                    std::string(
                        "Failed to start composed ExpertOverlay device maintenance: ") +
                    error.what());
            }
            catch (...)
            {
                return setError(
                    "Failed to start composed ExpertOverlay device maintenance with a non-standard exception");
            }
            return true;
        }
        if (!moe_expert_overlay_maintenance_service_)
            return true;
        if (moe_expert_overlay_maintenance_service_->state() !=
            MoEOverlayMaintenanceState::Prepared)
        {
            return setError(
                "ExpertOverlay host maintenance reached activation outside Prepared");
        }

        try
        {
            moe_expert_overlay_maintenance_service_->start();
        }
        catch (const std::exception &error)
        {
            return setError(
                std::string(
                    "Failed to start composed ExpertOverlay host maintenance: ") +
                error.what());
        }
        catch (...)
        {
            return setError(
                "Failed to start composed ExpertOverlay host maintenance with a non-standard exception");
        }
        return true;
    }

    void OrchestrationRunner::
        shutdownMoEExpertOverlayResidencyMaintenance(
            MoEOverlayMaintenanceDrainIntent intent) noexcept
    {
        if (intent == MoEOverlayMaintenanceDrainIntent::ResetComposition)
        {
            terminal_moe_optimization_status_.reset();
            terminal_moe_optimization_movement_ledger_.reset();
            terminal_moe_device_controller_drain_result_.reset();
        }

        for (const auto &binding : moe_pipeline_stage_bindings_)
            if (const auto &owner = binding->owners().host_maintenance; owner)
            {
                try { owner->stopAndDrain(); }
                catch (const std::exception &error)
                {
                    model_context_reuse_seal_succeeded_ = false;
                    model_context_reuse_seal_error_ = "CPU pipeline maintenance drain failed: " + std::string(error.what());
                    LOG_ERROR(model_context_reuse_seal_error_);
                }
            }
        /*
         * The topology-wide device service owns the only thread that may touch
         * descriptor inboxes or the physical slot ledger. Stop and join it
         * while both the inference executors and physical fabric are alive.
         * Terminal shutdown retains its completed authority evidence before
         * releasing the service; diagnostics belong to the runner lifecycle,
         * not to this implementation object's shorter resource lifetime.
         */
        const bool had_device_authority =
            static_cast<bool>(moe_overlay_device_controller_graph_service_);
        if (moe_overlay_device_controller_graph_service_)
        {
            auto device_drain_intent =
                MoEOverlayDeviceControllerDrainIntent::ReleaseResources;
            if (intent ==
                MoEOverlayMaintenanceDrainIntent::TerminalContextSeal)
            {
                // The exported reuse authority, not the mere use of Dynamic,
                // owns any obligation to restore loader-era placement. The
                // controller joins this local obligation across the topology.
                std::lock_guard<std::mutex> lock(model_context_reuse_mutex_);
                device_drain_intent = modelContextOverlayDrainIntent(
                    model_context_reuse_authority_, config_.moe_rebalance.mode);
            }
            terminal_moe_device_controller_drain_result_ =
                moe_overlay_device_controller_graph_service_->stopAndDrain(
                    device_drain_intent);
            if (intent ==
                MoEOverlayMaintenanceDrainIntent::TerminalContextSeal)
            {
                terminal_moe_optimization_status_ =
                    moe_overlay_device_controller_graph_service_
                        ->optimizationStatus();
                terminal_moe_optimization_movement_ledger_ =
                    moe_overlay_device_controller_graph_service_
                        ->movementLedger();
            }
        }
        moe_overlay_device_controller_graph_service_.reset();
        if (moe_expert_overlay_maintenance_service_)
        {
            try
            {
                moe_expert_overlay_maintenance_service_->stopAndDrain(
                    intent ==
                            MoEOverlayMaintenanceDrainIntent::
                                TerminalContextSeal
                        ? MoEOverlayMaintenanceDrainScope::
                              DistributedTopology
                        : MoEOverlayMaintenanceDrainScope::
                              ProcessLocalComposition);
                if (!moe_expert_overlay_maintenance_service_->healthy())
                {
                    LOG_ERROR(
                        "[OrchestrationRunner] ExpertOverlay maintenance stopped after failure: "
                        << moe_expert_overlay_maintenance_service_
                               ->failureMessage());
                }
            }
            catch (const std::exception &error)
            {
                LOG_ERROR(
                    "[OrchestrationRunner] ExpertOverlay maintenance drain failed: "
                    << error.what());
            }
            catch (...)
            {
                LOG_ERROR(
                    "[OrchestrationRunner] ExpertOverlay maintenance drain raised a non-standard exception");
            }

            if (intent ==
                    MoEOverlayMaintenanceDrainIntent::TerminalContextSeal &&
                !had_device_authority)
            {
                auto terminal_status =
                    moe_expert_overlay_maintenance_service_
                        ->optimizationStatus();
                if (!terminal_status.failed())
                {
                    terminal_status.state =
                        MoEOptimizationLifecycleState::Drained;
                    terminal_status.diagnostic.clear();
                    terminal_moe_optimization_status_ =
                        std::move(terminal_status);
                    terminal_moe_optimization_movement_ledger_ =
                        moe_expert_overlay_residency_authority_
                            ? moe_expert_overlay_residency_authority_
                                  ->movementLedger()
                            : MoEOptimizationMovementLedger{};
                }
            }
        }

        /* Dependency order is part of the asynchronous ownership contract. */
        moe_expert_overlay_maintenance_service_.reset();
        if (moe_expert_overlay_proposal_publisher_)
        {
            try
            {
                moe_expert_overlay_proposal_publisher_->stopAndDrain();
            }
            catch (const std::exception &error)
            {
                LOG_ERROR(
                    "[OrchestrationRunner] ExpertOverlay histogram publisher drain failed: "
                    << error.what());
            }
            catch (...)
            {
                LOG_ERROR(
                    "[OrchestrationRunner] ExpertOverlay histogram publisher drain raised a non-standard exception");
            }
        }
        moe_expert_overlay_proposal_publisher_.reset();

        if (intent == MoEOverlayMaintenanceDrainIntent::TerminalContextSeal &&
            preparedContextSealBoundary() == ModelContextPhysicalSealBoundary::LiveMutableResidency)
        {
            /*
             * The background scheduler is now quiescent, while transport
             * lanes, participant banks, physical slots, model registry, and
             * backend contexts are all still alive. This is the sole safe
             * interval for a reusable Dynamic context to return to its
             * prepared placement and atomically publish engine identities.
             *
             * Initialization also drains an empty/partial composition through
             * this function. Keeping that path outside this typed branch is
             * essential: declaring a context sealed before its residency
             * authority exists lets later slot recycling silently invalidate
             * loader-era logical expert bindings.
             */
            std::string reuse_error;
            if (!beginModelContextReuseSealIfNeeded(&reuse_error) ||
                !sealReusableModelContextPhysicalState(&reuse_error))
            {
                model_context_reuse_seal_succeeded_ = false;
                if (model_context_reuse_seal_error_.empty())
                {
                    model_context_reuse_seal_error_ = reuse_error.empty()
                        ? "prepared model context physical sealing failed"
                        : std::move(reuse_error);
                }
                LOG_ERROR(
                    "[OrchestrationRunner] "
                    << model_context_reuse_seal_error_);
            }
        }
        /* The maintenance/certification owner is quiescent. Retire the
         * controller before its transport dependencies; any reusable sealed
         * evidence was copied into the exported model contract beforehand. */
        moe_expert_overlay_economy_calibration_.reset();
        moe_expert_overlay_distributed_migration_transport_.reset();
        moe_expert_overlay_migration_transport_.reset();
        moe_expert_overlay_wave_factory_.reset();
        moe_expert_overlay_device_bank_publisher_.reset();
        moe_expert_overlay_physical_residency_fabric_.reset();
        moe_cpu_host_maintenance_.reset();
        moe_expert_overlay_remote_projection_transport_.reset();
        moe_expert_overlay_residency_consensus_.reset();
    }

    bool OrchestrationRunner::beginModelContextReuseSealIfNeeded(
        std::string *error) noexcept
    {
        if (error)
            error->clear();
        try
        {
            std::shared_ptr<ModelContextReuseAuthority> authority;
            {
                std::lock_guard<std::mutex> lock(
                    model_context_reuse_mutex_);
                authority = model_context_reuse_authority_;
            }
            if (!authority)
                return true;

            const auto state = authority->state();
            if (state == ModelContextReuseAuthority::State::Sealing)
                return true;
            if (state !=
                ModelContextReuseAuthority::State::RunnerExclusive)
            {
                if (error)
                {
                    *error =
                        "prepared model context reached teardown outside RunnerExclusive/Sealing";
                }
                return false;
            }
            return authority->beginSealing(error);
        }
        catch (const std::exception &exception)
        {
            if (error)
                *error = exception.what();
        }
        catch (...)
        {
            if (error)
            {
                *error =
                    "prepared model context sealing raised a non-standard exception";
            }
        }
        return false;
    }

    bool OrchestrationRunner::rebindModelRegistryFromReusableContextSeal(
        const MoEOverlayReusableContextSeal &seal,
        std::string *error,
        const std::shared_ptr<MoEOverlayParticipantResidencyRegistry> &residency) noexcept
    {
        if (error)
            error->clear();
        try
        {
            if (!seal.valid() || !model_ctx_ ||
                !residency)
            {
                if (error)
                {
                    *error =
                        "prepared model context rebind is missing a canonical physical seal, model, or participant-bank authority";
                }
                return false;
            }
            const auto weight_manager =
                model_ctx_->concreteWeightManager();
            if (!weight_manager)
            {
                if (error)
                {
                    *error =
                        "prepared model context rebind has no live weight registry";
                }
                return false;
            }

            /*
             * The physical seal is the sole terminal registry-rebind authority.
             * Host RCU snapshots remain deliberately stale when the device owns
             * placement, so consulting one here would recreate a host shadow and
             * compare its initial epoch with the device's restored durable epoch.
             */
            const auto participant_ids = residency->localParticipantIds();
            std::size_t scope_count = 0, resident_count = 0;
            for (const auto &bank : seal.local_banks)
            {
                scope_count += bank.layers.size();
                for (const auto &layer : bank.layers)
                    resident_count += std::count(layer.resident_mask.begin(), layer.resident_mask.end(), true);
            }
            std::string registry_error;
            if (!weight_manager->expertGemmRegistry()
                     .replaceParticipantResidencyFromSeal(seal, *residency, &registry_error))
            {
                if (error)
                {
                    *error = registry_error.empty()
                        ? "prepared model context registry replacement failed"
                        : std::move(registry_error);
                }
                return false;
            }
            PerfStatsCollector::addCounter(
                "moe_overlay_residency",
                "prepared_context_registry_rebinds",
                1.0,
                "model_teardown",
                {},
                {{"epoch", std::to_string(seal.source_epoch)},
                 {"local_participants",
                  std::to_string(participant_ids.size())},
                 {"participant_layer_scopes",
                  std::to_string(scope_count)},
                 {"resident_experts",
                  std::to_string(resident_count)},
                 {"compacted_shadow_experts",
                  std::to_string(seal.compacted_shadow_experts)}});
            return true;
        }
        catch (const std::exception &exception)
        {
            if (error)
                *error = exception.what();
        }
        catch (...)
        {
            if (error)
            {
                *error =
                    "prepared model context registry rebind raised a non-standard exception";
            }
        }
        return false;
    }

    ModelContextPhysicalSealBoundary OrchestrationRunner::preparedContextSealBoundary() const
    {
        if (!model_context_reuse_authority_)
            return ModelContextPhysicalSealBoundary::Unretained;
        std::vector<MoEOverlayMigrationStorageKind> storage;
        if (moe_pipeline_prepared_plan_)
        {
            for (const auto &stage : moe_pipeline_prepared_plan_->stages())
                storage.push_back(MoEOverlayCapacityAdmission::selectMigrationStorage(*stage.placement, config_.moe_rebalance.mode));
        }
        else if (config_.moe_routed_expert_plan)
            storage.push_back(MoEOverlayCapacityAdmission::selectMigrationStorage(*config_.moe_routed_expert_plan, config_.moe_rebalance.mode));
        return modelContextPhysicalSealBoundary(model_context_reuse_authority_, storage);
    }

    bool OrchestrationRunner::sealReusableModelContextPhysicalState(
        std::string *error) noexcept
    {
        if (error)
            error->clear();
        if (model_context_physical_state_sealed_)
            return model_context_reuse_seal_succeeded_;
        if (!model_context_reuse_authority_)
        {
            model_context_physical_state_sealed_ = true;
            return true;
        }

        try
        {
            if (preparedContextSealBoundary() == ModelContextPhysicalSealBoundary::RetiredImmutableSources)
            {
                if (runner_)
                    throw std::logic_error("Immutable prepared sources cannot be sealed before participant retirement");
                std::size_t resident_sources = 0;
                const auto verify_sources = [&](const std::shared_ptr<MoEOverlayParticipantResidencyRegistry> &residency)
                {
                    const auto weights = model_ctx_ ? model_ctx_->concreteWeightManager() : nullptr;
                    if (!residency || !weights)
                        throw std::logic_error("Immutable prepared source seal lost its model or participant registry");
                    resident_sources += residency->verifyImmutablePreparedSources(weights->expertGemmRegistry());
                };
                if (moe_pipeline_prepared_plan_)
                {
                    if (moe_pipeline_stage_bindings_.size() != moe_pipeline_prepared_plan_->stages().size())
                        throw std::logic_error("Immutable prepared source seal lost a pipeline stage");
                    for (const auto &binding : moe_pipeline_stage_bindings_)
                    {
                        if (!binding) throw std::logic_error("Immutable prepared source seal has an absent pipeline binding");
                        verify_sources(binding->owners().residency);
                    }
                }
                else if (config_.moe_routed_expert_plan && config_.moe_routed_expert_plan->usesExpertOverlayAuthority())
                {
                    if (!moe_expert_overlay_residency_authority_)
                        throw std::logic_error("Immutable prepared source seal lost its admitted residency authority");
                    verify_sources(moe_expert_overlay_participant_residency_);
                }
                model_context_physical_state_sealed_ = true;
                PerfStatsCollector::addCounter("moe_overlay_residency", "immutable_prepared_context_seals", 1.0,
                    "model_teardown", {}, {{"resident_sources", std::to_string(resident_sources)}});
                return true;
            }
            if (moe_pipeline_prepared_plan_)
            {
                if (moe_pipeline_stage_bindings_.size() != moe_pipeline_prepared_plan_->stages().size())
                    throw std::logic_error("Mutable pipeline sealing lost exact stage ownership");
                if (!model_context_pipeline_host_restored_)
                {
                    for (const auto &binding : moe_pipeline_stage_bindings_)
                        if (const auto &owner = binding->owners().host_maintenance; owner)
                        {
                            owner->stopAndDrain();
                            const auto seal = owner->restorePreparedPlacement();
                            if (!rebindModelRegistryFromReusableContextSeal(seal, error, binding->owners().residency)) return false;
                        }
                    model_context_pipeline_host_restored_ = true;
                }
                // Native directory sources remain immutable but may be bound by
                // captured children. Their final proof follows child retirement.
                if (runner_) return true;
                const auto weights = model_ctx_->concreteWeightManager();
                if (!weights) throw std::logic_error("Mutable pipeline seal lost its model weight registry");
                for (const auto &binding : moe_pipeline_stage_bindings_)
                {
                    if (binding->owners().host_maintenance)
                    {
                        if (!binding->owners().authority->initialPreparedPlacementPublished() ||
                            binding->owners().authority->activeTicketCount() != 0)
                            throw std::logic_error("CPU pipeline prepared placement changed after terminal restoration");
                    }
                    else (void)binding->owners().residency->verifyImmutablePreparedSources(weights->expertGemmRegistry());
                }
                model_context_physical_state_sealed_ = true;
                return true;
            }
            if (!moe_expert_overlay_residency_authority_)
            {
                const bool requires_dynamic_overlay_seal =
                    config_.moe_routed_expert_plan &&
                    config_.moe_routed_expert_plan
                        ->usesExpertOverlayAuthority() &&
                    config_.moe_rebalance.mode ==
                        MoERebalanceRuntimeMode::Dynamic;
                if (requires_dynamic_overlay_seal)
                {
                    if (error)
                    {
                        *error =
                            "Dynamic prepared-context sealing reached its terminal boundary without a residency authority";
                    }
                    return false;
                }
                model_context_physical_state_sealed_ = true;
                return true;
            }
            if (moe_cpu_host_maintenance_)
            {
                moe_cpu_host_maintenance_->stopAndDrain();
                const auto seal = moe_cpu_host_maintenance_->restorePreparedPlacement();
                if (!rebindModelRegistryFromReusableContextSeal(seal, error, moe_expert_overlay_participant_residency_)) return false;
                model_context_physical_state_sealed_ = true;
                return true;
            }
            if (!moe_expert_overlay_residency_authority_->migrationEnabled())
            {
                model_context_physical_state_sealed_ = true;
                return true;
            }
            if (!config_.moe_routed_expert_plan)
            {
                if (error)
                {
                    *error =
                        "Dynamic prepared-context reuse lost its frozen authority execution kind";
                }
                return false;
            }

            if (config_.moe_routed_expert_plan->authority_execution ==
                MoEOverlayAuthorityExecutionKind::DeviceResident)
            {
                if (!terminal_moe_device_controller_drain_result_ ||
                    terminal_moe_device_controller_drain_result_->intent !=
                        MoEOverlayDeviceControllerDrainIntent::
                            RestorePreparedContext ||
                    !terminal_moe_device_controller_drain_result_->valid())
                {
                    if (error)
                    {
                        *error =
                            "device-resident prepared-context sealing has no exact terminal restoration receipt";
                    }
                    return false;
                }
                if (moe_expert_overlay_residency_authority_
                        ->activeTicketCount() != 0u)
                {
                    if (error)
                    {
                        *error =
                            "device-resident prepared-context restoration completed while inference tickets remained live";
                    }
                    return false;
                }
                if (!moe_expert_overlay_physical_residency_fabric_)
                {
                    if (error)
                    {
                        *error =
                            "device-resident prepared-context sealing lost its physical residency fabric";
                    }
                    return false;
                }
                const auto physical_seal =
                    moe_expert_overlay_physical_residency_fabric_
                        ->sealReusableInitialPlacement(
                            terminal_moe_device_controller_drain_result_
                                ->durable_epoch,
                            error);
                if (!physical_seal ||
                    !rebindModelRegistryFromReusableContextSeal(
                        *physical_seal, error, moe_expert_overlay_participant_residency_))
                {
                    return false;
                }
                model_context_physical_state_sealed_ = true;
                PerfStatsCollector::addCounter(
                    "moe_overlay_residency",
                    "prepared_context_seals",
                    1.0,
                    "model_teardown",
                    {},
                    {{"restoration_waves",
                      std::to_string(
                          terminal_moe_device_controller_drain_result_
                              ->restoration_movement_waves)},
                     {"restored_epoch",
                      std::to_string(
                          terminal_moe_device_controller_drain_result_
                              ->durable_epoch)},
                     {"authority", "device"},
                     {"model_context_generation",
                      std::to_string(
                          model_context_reuse_authority_->generation())}});
                return true;
            }
            if (config_.moe_routed_expert_plan->authority_execution !=
                MoEOverlayAuthorityExecutionKind::HostResident)
            {
                if (error)
                {
                    *error =
                        "Dynamic prepared-context reuse has an unresolved authority execution kind";
                }
                return false;
            }
            if (moe_expert_overlay_residency_authority_
                    ->activeTicketCount() != 0u)
            {
                if (error)
                {
                    *error =
                        "prepared-context restoration began while inference tickets were still live";
                }
                return false;
            }

            IMoEOverlayResidencyTransport *transport =
                moe_expert_overlay_distributed_migration_transport_
                    ? static_cast<IMoEOverlayResidencyTransport *>(
                          moe_expert_overlay_distributed_migration_transport_
                              .get())
                    : static_cast<IMoEOverlayResidencyTransport *>(
                          moe_expert_overlay_migration_transport_.get());
            if (!transport ||
                !moe_expert_overlay_participant_residency_ ||
                !moe_expert_overlay_physical_residency_fabric_)
            {
                if (error)
                {
                    *error =
                        "prepared-context restoration is missing its transport, participant banks, or physical fabric";
                }
                return false;
            }

            const auto poll_until_idle = [&](std::string *poll_error)
            {
                const auto deadline = std::chrono::steady_clock::now() +
                    std::chrono::milliseconds(
                        collective_timeout_policy::
                            kDefaultCollectiveTimeoutMs);
                for (;;)
                {
                    const auto progress =
                        moe_expert_overlay_residency_authority_
                            ->advanceBackground();
                    if (!progress.ok())
                    {
                        if (poll_error)
                        {
                            *poll_error = progress.error.empty()
                                ? "prepared-context restoration background wave failed"
                                : progress.error;
                        }
                        return false;
                    }
                    if (progress.status ==
                            MoEOverlayResidencyApplyStatus::Idle)
                    {
                        return true;
                    }
                    if (std::chrono::steady_clock::now() >= deadline)
                    {
                        if (poll_error)
                        {
                            *poll_error =
                                "prepared-context restoration exceeded the canonical 30-second protocol deadline";
                        }
                        return false;
                    }
                    std::this_thread::yield();
                }
            };

            std::string progress_error;
            if (!poll_until_idle(&progress_error))
            {
                if (error)
                    *error = std::move(progress_error);
                return false;
            }

            std::size_t waves = 0u;
            const auto current =
                moe_expert_overlay_residency_authority_->snapshot();
            const std::size_t maximum_waves = current
                ? preparedPlacementRestorationWaveLimit(*current)
                : 0u;
            while (!moe_expert_overlay_residency_authority_
                        ->initialPreparedPlacementPublished())
            {
                if (maximum_waves == 0u || ++waves > maximum_waves)
                {
                    if (error)
                    {
                        *error =
                            "prepared-context restoration did not make bounded monotonic progress";
                    }
                    return false;
                }
                const auto transaction =
                    moe_expert_overlay_residency_authority_
                        ->proposeInitialPreparedPlacementRestoration();
                const auto started =
                    moe_expert_overlay_residency_authority_->beginApply(
                        transaction, *transport);
                if (started.status !=
                    MoEOverlayResidencyApplyStatus::Started)
                {
                    if (error)
                    {
                        *error = started.error.empty()
                            ? "prepared-context restoration wave did not start"
                            : started.error;
                    }
                    return false;
                }
                if (!poll_until_idle(&progress_error))
                {
                    if (error)
                        *error = std::move(progress_error);
                    return false;
                }
            }

            const auto published =
                moe_expert_overlay_residency_authority_->snapshot();
            if (!published || !published->valid())
            {
                if (error)
                {
                    *error =
                        "prepared-context physical sealing lost the restored published epoch";
                }
                return false;
            }
            const auto physical_seal =
                moe_expert_overlay_physical_residency_fabric_
                    ->sealReusableInitialPlacement(
                        published->epoch, error);
            if (!physical_seal ||
                !rebindModelRegistryFromReusableContextSeal(
                    *physical_seal, error, moe_expert_overlay_participant_residency_))
                return false;
            model_context_physical_state_sealed_ = true;
            PerfStatsCollector::addCounter(
                "moe_overlay_residency",
                "prepared_context_seals",
                1.0,
                "model_teardown",
                {},
                {{"restoration_waves", std::to_string(waves)},
                 {"model_context_generation",
                  std::to_string(
                      model_context_reuse_authority_->generation())}});
            return true;
        }
        catch (const std::exception &exception)
        {
            if (error)
                *error = exception.what();
        }
        catch (...)
        {
            if (error)
            {
                *error =
                    "prepared model context physical sealing raised a non-standard exception";
            }
        }
        return false;
    }

    bool OrchestrationRunner::sealModelContextDeviceMemoryRetention(
        std::vector<ModelDeviceMemoryRetention> *retention,
        std::string *error) const noexcept
    {
        if (error)
            error->clear();
        if (!retention)
        {
            if (error)
                *error = "model-context retention seal requires an output value";
            return false;
        }
        retention->clear();

        try
        {
            if (!model_ctx_)
            {
                if (error)
                    *error = "model-context retention seal lost its model owner";
                return false;
            }
            if (!reusable_execution_workspaces_ ||
                !reusable_execution_workspaces_->valid())
            {
                if (error)
                {
                    *error =
                        "model-context retention seal has no valid reusable workspace authority";
                }
                return false;
            }

            const auto weight_manager =
                model_ctx_->concreteWeightManager();
            if (!weight_manager)
            {
                if (error)
                    *error = "model-context retention seal has no WeightManager";
                return false;
            }

            const auto prepared_devices =
                rankLocalPreparedWeightAuthorityDevices(
                    plan_,
                    config_.moe_routed_expert_plan,
                    moe_expert_overlay_mpi_ctx_
                        ? moe_expert_overlay_mpi_ctx_
                        : mpi_ctx_);
            for (const DeviceId device : prepared_devices)
            {
                if (!device.is_gpu())
                    continue;
                if (weight_manager->preparedRecordCountForDevice(device) == 0u)
                {
                    if (error)
                    {
                        *error =
                            "model-context retention seal found no prepared records for " +
                            device.toString();
                    }
                    return false;
                }

                const size_t prepared_weight_bytes =
                    weight_manager->retainedPreparedDeviceBytes(device);
                const size_t reusable_workspace_bytes =
                    reusable_execution_workspaces_->retainedPrimaryBytes(
                        device);
                size_t immutable_table_bytes=model_ctx_->residentImmutableRowTableBytes(device);
                size_t immutable_lookup_staging_bytes=0;
                // Physical roots survive execution-reader teardown. A shared
                // reader may own remote cache roots, or no immutable roots at
                // all; neither case creates a source boundary on that device.
                const auto actual=model_ctx_->retainedAdaptiveImmutableLookupOwners({plan_.rank,device});
                if(!actual.empty() && immutable_table_bytes)
                    throw std::logic_error("Resident and adaptive immutable table ownership cannot overlap");
                for(const auto& owner:actual)
                {
                    if(owner.authority().get()!=physical_memory_authority_.get() ||
                       owner.authority().owner_before(physical_memory_authority_) || physical_memory_authority_.owner_before(owner.authority()) ||
                       owner.allocator()!=PhysicalMemoryAllocatorIdentity{plan_.rank,device})
                        throw std::invalid_argument("Adaptive retention must observe its exact original GPU/PMA root claims");
                    // Only actual disjoint surviving owners author these bytes.
                    // This projects their claims for retention; it never chooses
                    // capacity, reserves memory or copies cache occupancy.
                    if(owner.tableStorageBytes()>std::numeric_limits<std::size_t>::max()-immutable_table_bytes ||
                       owner.lookupStagingBytes()>std::numeric_limits<std::size_t>::max()-immutable_lookup_staging_bytes)
                        throw std::overflow_error("Adaptive immutable retention byte total overflowed");
                    immutable_table_bytes+=owner.tableStorageBytes();
                    immutable_lookup_staging_bytes+=owner.lookupStagingBytes();
                }
                ModelDeviceMemoryRetention row{
                    .device = device,
                    .prepared_weight_bytes = prepared_weight_bytes,
                    .reusable_workspace_bytes = reusable_workspace_bytes,
                    .immutable_table_bytes = immutable_table_bytes,
                    .immutable_lookup_staging_bytes = immutable_lookup_staging_bytes,
                };
                /*
                 * A participant that owns only model weights is a complete
                 * retention authority.  ExpertOverlay follower ranks can
                 * execute entirely from participant-local graph workspaces,
                 * leaving this runner's reusable primary-workspace registry
                 * empty.  ModelDeviceMemoryRetention deliberately models the
                 * four disjoint allocation classes independently, so requiring
                 * both to be positive would invalidate a real weights-only
                 * owner and make the next collective admission asymmetric.
                 */
                if (prepared_weight_bytes == 0u || !row.valid())
                {
                    if (error)
                    {
                        *error =
                            "model-context retention seal found invalid model-owned allocations for " +
                            device.toString() +
                            ": prepared_weight_bytes=" +
                            std::to_string(prepared_weight_bytes) +
                            " reusable_workspace_bytes=" +
                            std::to_string(reusable_workspace_bytes) +
                            " immutable_table_bytes=" + std::to_string(immutable_table_bytes);
                    }
                    return false;
                }
                retention->push_back(row);
                LOG_INFO(
                    "[ModelContextReuseContract] sealed retention device="
                    << device.toString()
                    << " prepared_weight_bytes="
                    << prepared_weight_bytes
                    << " reusable_workspace_bytes="
                    << reusable_workspace_bytes
                    << " immutable_table_bytes=" << immutable_table_bytes
                    << " total_bytes=" << row.totalBytes());
            }
            return true;
        }
        catch (const std::exception &exception)
        {
            if (error)
                *error = exception.what();
        }
        catch (...)
        {
            if (error)
            {
                *error =
                    "model-context retention sealing raised a non-standard exception";
            }
        }
        retention->clear();
        return false;
    }

    /**
     * @brief Publish one executable captured-prefill contract for ExpertOverlay.
     *
     * A sparse prefill transaction has to enter the same participant boundary
     * for every chunk. The contract freezes the root-selected bucket ladder
     * and, for a distributed world, reduces independent local capacities to
     * the one shape limit every endpoint can honor. This runs after memory
     * planning and before graph construction, so no captured graph or serial
     * compact arena can be bound to a larger uncoordinated request shape.
     */
    bool OrchestrationRunner::establishMoEOverlayPrefillScheduleContract()
    {
        plan_.runtime.overlay_prefill_schedule = {};

        const auto overlay_context =
            moe_expert_overlay_mpi_ctx_ ? moe_expert_overlay_mpi_ctx_ : mpi_ctx_;
        const auto overlay_execution = resolveOverlayExecutionPlanForRunner(
            config_.moe_routed_expert_plan,
            overlay_context);
        if ((!overlay_execution && !moe_pipeline_topology_) || !overlay_context)
        {
            return true;
        }
        const bool distributed = overlay_context->world_size() > 1;
        if (distributed && overlay_context->communicator() == MPI_COMM_NULL)
        {
            return setError(
                "Distributed MoE ExpertOverlay requires a live MPI "
                "communicator to publish its prefill schedule contract");
        }

        const int local_graph_capacity = plan_.runtime.resident_graph_rows;
        const int requested_segment_rows =
            plan_.runtime.moe_routed_prefill.overlay_segment_rows;
        if (local_graph_capacity <= 0 || requested_segment_rows <= 0)
        {
            return setError(
                "MoE ExpertOverlay rank has no positive planner-admitted graph "
                "capacity or configured prefill segment size");
        }
        const int local_capacity =
            std::min(local_graph_capacity, requested_segment_rows);

        int common_capacity = local_capacity;
        if (distributed &&
            MPI_Allreduce(
                &local_capacity,
                &common_capacity,
                1,
                MPI_INT,
                MPI_MIN,
                overlay_context->communicator()) != MPI_SUCCESS)
        {
            return setError(
                "Distributed MoE ExpertOverlay could not reduce its jointly "
                "admitted prefill segment capacity");
        }
        if (common_capacity <= 0)
        {
            return setError(
                "MoE ExpertOverlay resolved a non-positive prefill segment capacity");
        }

        const int continuation_root_rank =
            overlay_execution ? overlay_execution->continuation_root_rank :
            resolveMoEOverlayRequestAuthorityRank(nullptr, moe_pipeline_prepared_plan_, overlay_context).value_or(plan_.rank);
        if (continuation_root_rank < 0 ||
            continuation_root_rank >= overlay_context->world_size())
        {
            return setError(
                "Distributed MoE ExpertOverlay has an invalid continuation "
                "root rank for prefill schedule publication");
        }

        std::vector<int> bucket_rows;
        int bucket_count = 0;
        if (!distributed ||
            overlay_context->rank() == continuation_root_rank)
        {
            const int configured_cache_entries =
                debugEnv().execution.prefill_graph_max_cached_buckets;
            const bool exact = plan_.runtime.ngram_table_placement && plan_.runtime.ngram_table_placement->fixedHeadPipeline();
            if (!exact && configured_cache_entries <=
                static_cast<int>(
                    kExpertOverlayPrefillGraphIdentityReserve))
            {
                return setError(
                    "MoE ExpertOverlay prefill graph cache cannot retain its runtime identities and one bucket");
            }
            if (exact) {
                (void)exactPrefillGraphCacheEntries(common_capacity,
                    debugEnv().execution.prefill_graph_max_cached_buckets_explicit
                        ? std::optional{configured_cache_entries} : std::nullopt);
                bucket_rows = exactPrefillGraphWidths(common_capacity);
            } else bucket_rows = retainedRawPrefillGraphBucketLadder(
                debugEnv().execution.prefill_graph_bucket_sizes,
                common_capacity,
                debugEnv().execution.prefill_graph_min_seq,
                static_cast<std::size_t>(configured_cache_entries) -
                    kExpertOverlayPrefillGraphIdentityReserve);
            if (bucket_rows.empty() ||
                bucket_rows.size() >
                    static_cast<size_t>(std::numeric_limits<int>::max()))
            {
                return setError(
                    "Distributed MoE ExpertOverlay root could not publish a "
                    "bounded prefill bucket ladder");
            }
            bucket_count = static_cast<int>(bucket_rows.size());
        }

        if (distributed &&
            MPI_Bcast(
                &bucket_count,
                1,
                MPI_INT,
                continuation_root_rank,
                overlay_context->communicator()) != MPI_SUCCESS)
        {
            return setError(
                "Distributed MoE ExpertOverlay failed to publish its prefill "
                "bucket count");
        }
        if (bucket_count <= 0)
        {
            return setError(
                "MoE ExpertOverlay resolved an empty prefill bucket contract");
        }

        if (distributed &&
            overlay_context->rank() != continuation_root_rank)
        {
            bucket_rows.resize(static_cast<size_t>(bucket_count));
        }
        if (distributed &&
            MPI_Bcast(
                bucket_rows.data(),
                bucket_count,
                MPI_INT,
                continuation_root_rank,
                overlay_context->communicator()) != MPI_SUCCESS)
        {
            return setError(
                "Distributed MoE ExpertOverlay failed to publish its prefill "
                "bucket ladder");
        }

        const auto normalized_bucket_rows =
            normalizePrefillGraphBuckets(bucket_rows);
        if (normalized_bucket_rows != bucket_rows ||
            normalized_bucket_rows.empty() ||
            normalized_bucket_rows.back() > common_capacity)
        {
            return setError(
                "Distributed MoE ExpertOverlay received an invalid prefill "
                "bucket ladder");
        }

        plan_.runtime.overlay_prefill_schedule = {
            .graph_row_capacity = common_capacity,
            .bucket_rows = std::move(bucket_rows),
        };
        /*
         * Graph builders consume the resolved segment size through the normal
         * typed runner configuration. Publishing the reduced value here keeps
         * root and endpoint arenas identical even when one rank admitted a
         * smaller dense graph than its peers.
         */
        plan_.runtime.moe_routed_prefill.overlay_segment_rows =
            common_capacity;

        PerfStatsCollector::addCounter(
            "forward_graph",
            "moe_overlay_prefill_schedule_contract_rows",
            static_cast<double>(common_capacity),
            "model_setup",
            {},
            {
                {"authority", "continuation_root"},
                {"continuation_root_rank", std::to_string(continuation_root_rank)},
                {"local_graph_capacity", std::to_string(local_graph_capacity)},
                {"local_segment_capacity", std::to_string(local_capacity)},
                {"requested_segment_rows", std::to_string(requested_segment_rows)},
                {"bucket_count", std::to_string(bucket_count)},
                {"distributed", distributed ? "true" : "false"},
                {"immutable", "true"},
            });
        return true;
    }

    bool OrchestrationRunner::initializeMoEOverlayRankBatchTransports()
    {
        moe_overlay_rank_batch_transport_registry_.reset();
        if(moe_pipeline_prepared_plan_) {
            try {
                moe_pipeline_stage_bindings_=moe_pipeline_prepared_plan_->prepareOriginalCPUStageTransports(
                    model_ctx_->concreteLoader(),mpi_ctx_,cluster_inventory_,plan_.runtime,moe_pipeline_stage_bindings_);
                return true;
            }
            catch(const std::exception& error) {
                return setError(std::string("Original CPU pipeline stage transport setup failed: ")+error.what());
            }
        }


        const auto overlay_context =
            moe_expert_overlay_mpi_ctx_ ? moe_expert_overlay_mpi_ctx_ : mpi_ctx_;
        const auto overlay_execution = resolveOverlayExecutionPlanForRunner(
            config_.moe_routed_expert_plan,
            overlay_context);
        if (!overlay_execution || !overlay_context)
        {
            return true;
        }
        // Preserve the existing no-channel lifecycle for continuation-local
        // GPUs and a same-rank CPU tier. The one typed planner predicate adds
        // channels only for genuine off-continuation GPU/rank endpoints.
        if (config_.moe_routed_expert_plan &&
            !MoEOverlayActivationChannelPlanner::requiresActivationChannels(*config_.moe_routed_expert_plan))
            return true;
        if (overlay_context->world_size() > 1 && overlay_context->communicator() == MPI_COMM_NULL)
        {
            return setError(
                "Distributed MoE ExpertOverlay cannot pre-create node-local "
                "activation channels without a live MPI communicator");
        }
        if (!model_ctx_ || !config_.moe_routed_expert_plan ||
            !moe_expert_overlay_residency_authority_)
        {
            return setError(
                "Distributed MoE ExpertOverlay activation-channel preflight "
                "requires the loaded model, frozen placement, and residency authority");
        }

        try
        {
            const auto snapshot =
                moe_expert_overlay_residency_authority_->snapshot();
            if (!snapshot || !snapshot->valid() ||
                !snapshot->placement_plan)
            {
                return setError(
                    "Distributed MoE ExpertOverlay activation-channel preflight "
                    "received no valid frozen owner-map snapshot");
            }
            const auto &placement = *snapshot->placement_plan;
            const auto &owner_map = snapshot->originalOwnerMap();
            const MTPRuntimeConfig &active_mtp = retainedMTPConfig();
            const bool retains_mtp_capacity =
                retainsMTPGraphCapacity(active_mtp);
            const auto metadata =
                resolveMoERoutedExpertModelMetadataForModel(
                    *model_ctx_, active_mtp);
            const int top_k = model_ctx_->concreteLoader().getInt(
                model_ctx_->architecture() + ".expert_used_count", 0);
            const int max_decode_rows =
                retains_mtp_capacity
                    ? std::max(
                          1,
                          resolveMTPRetainedTargetQueryRows(active_mtp))
                    : 1;
            const int max_mtp_draft_depth =
                resolveMTPRetainedDraftCapacity(active_mtp);
            const int max_graph_rows = std::max(
                plan_.runtime.overlay_prefill_schedule.graph_row_capacity,
                max_decode_rows);
            if (metadata.d_model <= 0 || top_k <= 0 || max_graph_rows <= 0)
            {
                return setError(
                    "Distributed MoE ExpertOverlay activation-channel preflight "
                    "resolved invalid model or graph geometry");
            }

            const auto graph_family =
                resolveMoEOverlayInferenceGraphFamilyIdentity(
                    model_ctx_->concreteLoader(),
                    model_ctx_->architecture(),
                    model_ctx_->totalBlockCount(),
                    retains_mtp_capacity
                        ? MoEOverlayMTPGraphFamilyPolicy::
                              RetainModelSidecars
                        : MoEOverlayMTPGraphFamilyPolicy::MainOnly,
                    /*graph_family_generation=*/1,
                    max_graph_rows,
                    max_decode_rows,
                    std::max(1, plan_.runtime.batch_size),
                    max_mtp_draft_depth).forRoutedLayerInterval(
                        placement.first_model_layer,
                        placement.placementLayerCapacity());
            moe_overlay_rank_batch_transport_registry_=prepareOriginalMoEOverlayRankBatchTransports({
                .placement=placement,.owner_map=owner_map,.inventory=cluster_inventory_,.graph_family=graph_family,
                .original_context=overlay_context,.physical_memory_authority=physical_memory_authority_,
                .activation_batch_inventory=moe_expert_overlay_memory_admission_?moe_expert_overlay_memory_admission_->activation_batch_inventory:nullptr,
                .original_cpu_joint_owners=snapshot->jointOwnerMap(),
                .d_model=metadata.d_model,.top_k=top_k});
            return true;
        }
        catch (const std::exception &error)
        {
            return setError(
                std::string(
                    "Failed to pre-create MoE ExpertOverlay rank-batch transports: ") +
                error.what());
        }
    }

    bool OrchestrationRunner::
        materializeOrdinaryServingGraphFamilyWithoutLaunch()
    {
        if (!runner_)
        {
            return setError(
                "Ordinary serving graph setup requires a constructed inference runner");
        }

        const auto &overlay_plan = config_.moe_routed_expert_plan;
        if (moe_pipeline_prepared_plan_ || (overlay_plan && overlay_plan->usesExpertOverlayAuthority()))
        {
            /*
             * Overlay graph capture is inseparable from its exact background
             * transfer epoch. Its maintenance composer has already sealed the
             * same runner through the overlay-specific setup transition.
             */
            return true;
        }

        const ServingGraphPreparationKind preparation_kind =
            runner_->servingGraphPreparationKind();
        if (preparation_kind == ServingGraphPreparationKind::EagerHostGraph &&
            !plan_.usesLocalPP())
            return true;
        if (preparation_kind == ServingGraphPreparationKind::Unresolved)
        {
            if (!runner_->primaryDeviceId().is_gpu() ||
                !debugEnv().execution.gpu_graphs)
            {
                return true;
            }
            return setError(
                "GPU runner did not declare a serving graph preparation lifecycle");
        }

        const auto &execution = debugEnv().execution;
        // Setup and request scheduling use the same raw-prompt ladder as
        // ForwardExecutionEngine preflight. Merely clipping to arena capacity
        // admitted a 64-row graph for a 39-token HTTP request even though the
        // padded-bucket floor was 256, leaving an admitted input bank stranded
        // when preflight rejected execution. Exact verifier/decode geometries
        // remain independent of this raw-prompt coalescing policy.
        // CPU-only pipelines also deferred workspace declaration until their
        // ingress owners existed. They seal that host family at its admitted
        // row capacity; no GPU bucket ladder or native executable is implied.
        const bool host_pipeline =
            preparation_kind == ServingGraphPreparationKind::EagerHostGraph;
        const std::vector<int> buckets = host_pipeline
            ? std::vector<int>{plan_.runtime.resident_graph_rows}
            : rawPrefillGraphBucketsForResidentCapacity(
                execution.prefill_graph_bucket_sizes,
                plan_.runtime.resident_graph_rows,
                execution.prefill_graph_min_seq);
        ServingGraphFamilyMaterializationPlan family_plan{
            .prefill_bucket_rows = buckets,
            .prefill_pad_token_id = execution.prefill_graph_pad_token_id,
            .main_decode_graph =
                ServingMainDecodeGraphKind::HistoryBearingSerial,
        };
        if (!family_plan.valid() ||
            (!host_pipeline && (!execution.gpu_graphs || !execution.prefill_graph_buckets)))
        {
            return setError(
                "Serving graph setup has no valid memory-admitted prefill family");
        }
        if (!runner_->materializeServingGraphFamilyWithoutLaunch(family_plan))
        {
            return setError(
                "Ordinary runner could not seal its admitted serving graph family");
        }

        PerfStatsCollector::addCounter(
            "forward_graph",
            "ordinary_serving_family_completions",
            1.0,
            "setup",
            runner_->primaryDeviceId().toString(),
            {{"prefill_buckets", std::to_string(buckets.size())},
             {"graph_row_capacity",
              std::to_string(plan_.runtime.resident_graph_rows)},
             {"main_decode", "history_bearing_serial"},
             {"preparation_kind", host_pipeline ? "eager_host_graph" : "native_device_executable_family"},
             {"request_state_mutations", "0"},
             {"executable_launches", "0"}});
        return true;
    }

    bool OrchestrationRunner::applyConfiguredSnapshotCaptureSetup()
    {
        if (!snapshot_capture_setup_.enabled())
            return true;
        if (!runner_)
        {
            return setError(
                "Snapshot capture setup requires a constructed inference runner");
        }

        /*
         * Install the filter while capture is still disabled, then enable the
         * callback exactly once. DeviceGraphOrchestrator consequently records
         * one topology epoch containing the final set of D2D copy nodes; it
         * never captures an unfiltered intermediate graph.
         */
        runner_->setSnapshotCaptureFilter(snapshot_capture_setup_.filter);
        runner_->enableSnapshotCapture(
            snapshot_capture_setup_.output_directory);
        PerfStatsCollector::addCounter(
            "forward_graph",
            "snapshot_topology_installed_before_serving_capture",
            1.0,
            "setup",
            runner_->primaryDeviceId().toString(),
            {{"filter_mode",
              snapshot_capture_setup_.filter.empty() ? "all" : "selected"},
             {"selected_keys",
              std::to_string(snapshot_capture_setup_.filter.size())}});
        return true;
    }

    bool OrchestrationRunner::
        materializeCurrentServingGraphFamilyVariantWithoutLaunch()
    {
        if (!runner_)
        {
            return setError(
                "Serving graph variant materialization requires a constructed inference runner");
        }

        const auto &overlay_plan = config_.moe_routed_expert_plan;
        if (!moe_pipeline_prepared_plan_ && (!overlay_plan || !overlay_plan->usesExpertOverlayAuthority()))
            return materializeOrdinaryServingGraphFamilyWithoutLaunch();

        const auto overlay_context =
            moe_expert_overlay_mpi_ctx_ ? moe_expert_overlay_mpi_ctx_
                                        : mpi_ctx_;
        const auto execution = resolveOverlayExecutionPlanForRunner(
            overlay_plan, overlay_context);
        if ((!execution && !moe_pipeline_prepared_plan_) || !overlay_context)
        {
            return setError(
                "ExpertOverlay serving graph variant lost its rank execution plan");
        }

        const auto &schedule = plan_.runtime.overlay_prefill_schedule;
        ServingGraphFamilyMaterializationPlan family_plan{
            .prefill_bucket_rows = schedule.bucket_rows,
            .prefill_pad_token_id =
                debugEnv().execution.prefill_graph_pad_token_id,
            .main_decode_graph =
                ServingMainDecodeGraphKind::HistoryBearingSerial,
        };
        if (!schedule.enabled() || !family_plan.valid() ||
            schedule.graph_row_capacity <= 0 ||
            family_plan.prefill_bucket_rows.back() >
                schedule.graph_row_capacity)
        {
            return setError(
                "ExpertOverlay diagnostic graph variant disagrees with the frozen prefill schedule");
        }
        bool materialized = false;
        if (moe_pipeline_topology_ && moe_pipeline_topology_->globalPipeline())
        {
            const auto weights = model_ctx_ ? model_ctx_->concreteWeightManager() : nullptr;
            if (!weights)
                return setError("Global pipeline serving graph has no original WeightManager authority");
            // Nested TP children deliberately retain shared sources. Only
            // this original rank's complete global parent may close that gate.
            materialized = materializePreparedPipelineGraphFamily(*runner_, *weights, family_plan);
        }
        else
            materialized = runner_->materializeServingGraphFamilyWithoutLaunch(family_plan);
        if (!materialized)
        {
            return setError(
                "ExpertOverlay runner could not materialize the selected serving graph variant");
        }
        return true;
    }

    bool OrchestrationRunner::
        deactivatePreparedSnapshotCaptureAndMaterializeLeanServingFamily()
    {
        if (snapshot_capture_setup_.mode !=
            SnapshotCaptureSetup::Mode::PreparedInactive)
        {
            return true;
        }
        if (!runner_)
        {
            return setError(
                "Prepared-inactive snapshot setup has no constructed inference runner");
        }

        /*
         * Delegate directly so SnapshotCaptureSetup remains the authority for the
         * later diagnostic activation. DeviceGraphOrchestrator maps disabled
         * snapshots back to the stable lean identity rather than inventing a new
         * monotonic cache generation.
         */
        runner_->disableSnapshotCapture();
        if (!materializeCurrentServingGraphFamilyVariantWithoutLaunch())
            return false;

        prepared_inactive_snapshot_capture_ready_ = true;
        PerfStatsCollector::addCounter(
            "forward_graph",
            "prepared_inactive_snapshot_families",
            1.0,
            "setup",
            runner_->primaryDeviceId().toString(),
            {{"diagnostic_variant", "materialized_unlaunched"},
             {"lean_variant", "materialized_unlaunched"},
             {"active_variant", "lean"}});
        return true;
    }

    bool OrchestrationRunner::
        bindMoEOverlayTransferProgressAndMaterializeServingGraphFamily()
    {
        const auto overlay_context =
            moe_expert_overlay_mpi_ctx_ ? moe_expert_overlay_mpi_ctx_
                                        : mpi_ctx_;
        const auto execution = resolveOverlayExecutionPlanForRunner(
            config_.moe_routed_expert_plan,
            overlay_context);
        if (!execution)
        {
            return true;
        }
        if (!overlay_context)
        {
            return setError(
                "ExpertOverlay serving graph setup lost its MPI/rank context");
        }
        if (!runner_)
        {
            return setError(
                "ExpertOverlay serving graph setup requires a built inference runner");
        }

        /*
         * The physical fabric is the sole command/byte authority, while each
         * exact device runner is the sole owner of its serving-graph cache.
         * Join those two setup-stable identities before capture.  The resulting
         * cache-private auxiliary branch claims commands at graph entry and
         * progresses them on the epoch's independent stream in parallel with
         * model work.  The fabric also retains its standalone submission path
         * for commands published after the final inference boundary; neither
         * path creates a second movement or placement authority.
         */
        std::size_t transfer_progress_epoch_count = 0u;
        if (moe_expert_overlay_physical_residency_fabric_)
        {
            try
            {
                for (const DeviceId device :
                     moe_expert_overlay_physical_residency_fabric_
                         ->transferProgressDevices())
                {
                    auto epoch =
                        moe_expert_overlay_physical_residency_fabric_
                            ->transferProgressEpoch(device);
                    if (!epoch || epoch->device() != device ||
                        !runner_->installMoEOverlayTransferProgressEpoch(
                            std::move(epoch)))
                    {
                        return setError(
                            "ExpertOverlay could not bind the physical transfer-progress epoch to its exact inference runner on " +
                            device.toString());
                    }
                    ++transfer_progress_epoch_count;
                }
            }
            catch (const std::exception &error)
            {
                return setError(
                    std::string(
                        "ExpertOverlay transfer-progress inventory binding failed: ") +
                    error.what());
            }
        }

        const ServingGraphPreparationKind preparation_kind =
            runner_->servingGraphPreparationKind();
        if (preparation_kind ==
            ServingGraphPreparationKind::Unresolved)
        {
            return setError(
                "ExpertOverlay runner did not declare a serving graph preparation lifecycle");
        }
        if (preparation_kind ==
                ServingGraphPreparationKind::EagerHostGraph &&
            transfer_progress_epoch_count != 0u)
        {
            return setError(
                "ExpertOverlay eager host graph unexpectedly owns a device transfer-progress epoch");
        }

        const auto &schedule = plan_.runtime.overlay_prefill_schedule;
        ServingGraphFamilyMaterializationPlan family_plan{
            .prefill_bucket_rows = schedule.bucket_rows,
            .prefill_pad_token_id =
                debugEnv().execution.prefill_graph_pad_token_id,
            .main_decode_graph =
                ServingMainDecodeGraphKind::HistoryBearingSerial,
        };
        if (!schedule.enabled() || !family_plan.valid() ||
            schedule.graph_row_capacity <= 0 ||
            family_plan.prefill_bucket_rows.back() >
                schedule.graph_row_capacity)
        {
            return setError(
                "ExpertOverlay serving graph setup requires the frozen common prefill schedule produced by memory admission");
        }

        /*
         * Preparation kind selects the work performed by the runner; it does
         * not replace the common admission transition. A native participant
         * captures every executable here, while an eager CPU follower proves
         * that its retained endpoints match the frozen family and changes its
         * ticket lifecycle from Built to Sealed. Skipping this call for an
         * eager follower left that typed state at Built even though the
         * surrounding rank-consensus phase succeeded, so transaction zero was
         * guaranteed to be rejected. Every overlay participant must cross the
         * same setup edge before request-ticket authority can be installed.
         */
        if (!runner_->materializeServingGraphFamilyWithoutLaunch(family_plan))
        {
            return setError(
                "ExpertOverlay runner could not seal its admitted serving graph family");
        }
        PerfStatsCollector::addCounter(
            "forward_graph",
            "overlay_serving_family_completions",
            1.0,
            "setup",
            "rank",
            {{"prefill_buckets",
              std::to_string(family_plan.prefill_bucket_rows.size())},
             {"graph_row_capacity",
              std::to_string(schedule.graph_row_capacity)},
             {"main_decode", "history_bearing_serial"},
             {"preparation_kind",
              preparation_kind ==
                      ServingGraphPreparationKind::
                          NativeDeviceExecutableFamily
                  ? "native_device_executable_family"
                  : "eager_host_graph"},
             {"runner_role",
              toString(execution->currentRankPlan().execution_kind)},
             {"rank_scope",
              overlay_context->world_size() > 1
                  ? "cross_rank"
                  : "rank_local"},
             {"transfer_progress_binding",
              transfer_progress_epoch_count != 0u
                  ? "graph_owned_parallel_branch"
                  : "none"},
             {"transfer_progress_epochs",
              std::to_string(transfer_progress_epoch_count)},
             {"capacity_authority", "overlay_prefill_schedule"},
             {"ticket_authority_installed", "false"}});
        return true;
    }

    bool OrchestrationRunner::initializeMoEOverlayInferenceTransactions()
    {
        const auto overlay_context =
            moe_expert_overlay_mpi_ctx_ ? moe_expert_overlay_mpi_ctx_
                                        : mpi_ctx_;
        const auto execution = resolveOverlayExecutionPlanForRunner(
            config_.moe_routed_expert_plan, overlay_context);
        auto *rank_runner = dynamic_cast<RankOrchestrator *>(runner_.get());
        const auto local_endpoints = rank_runner ? rank_runner->rankLocalExpertEndpoints() : nullptr;
        if (!execution || !overlay_context ||
            (overlay_context->world_size() <= 1 && !local_endpoints))
        {
            return true;
        }
        const auto &local_rank_plan = execution->currentRankPlan();
        const int source_rank = execution->continuation_root_rank;
        const int local_rank = overlay_context->rank();
        const std::size_t expert_follower_count =
            static_cast<std::size_t>(std::count_if(
                execution->rank_plans.begin(),
                execution->rank_plans.end(),
                [](const OverlayRankPlan &rank_plan)
                {
                    return rank_plan.usesExpertTransactionFollower();
                })) + (local_endpoints ? 1u : 0u);
        if (source_rank < 0 ||
            source_rank >= overlay_context->world_size() ||
            (local_rank == source_rank) !=
                local_rank_plan.ownsTransactionAuthority())
        {
            return setError(
                "Distributed ExpertOverlay transaction lifecycle disagrees "
                "with its continuation command authority");
        }
        if (moe_overlay_inference_transaction_coordinator_ ||
            moe_overlay_inference_transaction_follower_)
        {
            return setError(
                "ExpertOverlay inference transaction authority is setup-only");
        }

        const auto record_transaction_control_plane =
            [&](const char *control_plane)
        {
            PerfStatsCollector::addCounter(
                "moe_overlay_transaction",
                "control_plane_initializations",
                1.0,
                "graph_setup",
                {},
                {{"execution_kind",
                  toString(local_rank_plan.execution_kind)},
                 {"control_plane", control_plane},
                 {"source_rank", std::to_string(source_rank)},
                 {"local_rank", std::to_string(local_rank)},
                 {"expert_followers",
                  std::to_string(expert_follower_count)}});
        };

        /*
         * Dense NodeTP peers already follow the ordinary coordinated model
         * command and execute their authenticated rank-batch target stages
         * inside that full graph.  A retained-graph ticket follower is a
         * different, mutually exclusive lifecycle reserved for ranks that own
         * sparse expert endpoints and no continuation graph.  When no such
         * ranks exist, the continuation authority also needs no transaction
         * coordinator: the normal command plus rank-batch protocol is the
         * complete production transaction.
         */
        switch (local_rank_plan.execution_kind)
        {
        case OverlayRankExecutionKind::ContinuationPeer:
            record_transaction_control_plane(
                "coordinated_dense_command_peer");
            return true;
        case OverlayRankExecutionKind::RelayOnly:
            record_transaction_control_plane("inactive_relay");
            return true;
        case OverlayRankExecutionKind::ContinuationAuthority:
            if (expert_follower_count == 0u)
            {
                record_transaction_control_plane(
                    "coordinated_dense_command_authority");
                return true;
            }
            break;
        case OverlayRankExecutionKind::ExpertOnlyFollower:
            if (!local_rank_plan.loads_expert_weights)
            {
                return setError(
                    "ExpertOverlay transaction follower owns no expert weights");
            }
            break;
        }

        if (!runner_ || !model_ctx_ ||
            !moe_expert_overlay_residency_authority_)
        {
            return setError(
                "Distributed ExpertOverlay retained-graph transactions "
                "require a built runner, model context, and residency authority");
        }

        /*
         * Residency authority construction intentionally precedes graph and
         * transaction construction. The probe therefore has model lifetime,
         * but no runner binding: the root transaction coordinator owns exact
         * heterogeneous prefill terminals, orchestration owns root decode/MTP
         * intervals, and the follower owns its authenticated remote interval.
         * This keeps measurement ownership independent of whether the root is
         * one DeviceGraphOrchestrator or a flat LocalTP RankOrchestrator.
         */
        if (config_.moe_rebalance.mode ==
            MoERebalanceRuntimeMode::Dynamic)
        {
            if (!moe_expert_overlay_interference_probe_)
            {
                return setError(
                    "Distributed Dynamic ExpertOverlay lost its model-lifetime inference calibration probe");
            }
        }

        const auto snapshot =
            moe_expert_overlay_residency_authority_->snapshot();
        if (!snapshot || !snapshot->valid())
        {
            return setError(
                "Distributed ExpertOverlay inference transactions require a "
                "valid initial residency snapshot");
        }

        const MTPRuntimeConfig &active_mtp = retainedMTPConfig();
        const bool retains_mtp_capacity =
            retainsMTPGraphCapacity(active_mtp);
        const int max_mtp_draft_depth =
            resolveMTPRetainedDraftCapacity(active_mtp);
        const int max_decode_rows =
            retains_mtp_capacity
                ? resolveMTPRetainedTargetQueryRows(active_mtp)
                : 1;
        const int max_graph_rows = std::max(
            plan_.runtime.overlay_prefill_schedule.graph_row_capacity,
            max_decode_rows);
        const int max_request_count =
            std::max(1, plan_.runtime.batch_size);
        const auto graph_family =
            resolveMoEOverlayInferenceGraphFamilyIdentity(
                model_ctx_->concreteLoader(),
                model_ctx_->architecture(),
                model_ctx_->totalBlockCount(),
                retains_mtp_capacity
                    ? MoEOverlayMTPGraphFamilyPolicy::RetainModelSidecars
                    : MoEOverlayMTPGraphFamilyPolicy::MainOnly,
                /*graph_family_generation=*/1,
                max_graph_rows,
                max_decode_rows,
                max_request_count,
                max_mtp_draft_depth).forRoutedLayerInterval(
                    config_.moe_routed_expert_plan->first_model_layer,
                    config_.moe_routed_expert_plan->placementLayerCapacity());
        const std::size_t slot_count =
            moeOverlayInferenceTransactionSlotCount(
                max_mtp_draft_depth);
        if (source_rank < 0 ||
            source_rank >= overlay_context->world_size() ||
            slot_count == 0)
        {
            return setError(
                "Distributed ExpertOverlay inference resolved an invalid "
                "source rank or retained transaction capacity");
        }

        /*
         * Prefill already crosses a typed heterogeneous transaction boundary
         * for each retained bucket. Only the continuation authority accumulates
         * its authenticated real-row count. The next decode/forced-token command
         * sidebands that total to every rank in its existing payload, preventing
         * follower-local capture schedules from becoming admission authority.
         */
        MoEOverlayRetiredPrefillProgressSink
            retired_prefill_progress_sink;
        if (config_.moe_rebalance.mode ==
                MoERebalanceRuntimeMode::Dynamic &&
            moe_overlay_device_controller_graph_service_ &&
            local_rank == source_rank)
        {
            retired_prefill_progress_sink =
                [this](
                    std::uint64_t completed_tokens,
                    std::string *error) -> bool
            {
                if (error)
                    error->clear();
                if (completed_tokens == 0u)
                    return true;

                auto &pending =
                    moe_overlay_pending_prefill_progress_tokens_;
                std::uint64_t observed = pending.load(
                    std::memory_order_relaxed);
                const std::uint64_t payload_max =
                    static_cast<std::uint64_t>(
                        std::numeric_limits<std::int32_t>::max());
                while (observed <= payload_max &&
                       completed_tokens <= payload_max - observed)
                {
                    if (pending.compare_exchange_weak(
                            observed,
                            observed + completed_tokens,
                            std::memory_order_release,
                            std::memory_order_relaxed))
                    {
                        return true;
                    }
                }
                if (error)
                {
                    *error =
                        "Continuation retired-prefill progress exceeds the typed command sideband";
                }
                return false;
            };
        }

        /*
         * A hosted generation command may retire hundreds of complete ordinary
         * or MTP graph sequences before returning its one terminal response.
         * Feed their device-authenticated committed-token deltas directly into
         * each rank's one process-local maintenance authority. The continuation
         * coordinator receives the frontier from the scheduler ticket; expert
         * followers receive the same frontier on their already-required
         * transaction ticket. The callback is deliberately a wake-only edge:
         * placement policy, MPI, transfer, and publication stay on the
         * background authority. The topology-wide device authority retains a
         * terminal acknowledgement only on the continuation rank because its
         * followers have no outer decoder. The older host authority retains
         * the acknowledgement on every process: its follower worker already
         * consumes that exact outer boundary after the hosted command returns.
         */
        MoEOverlayRetiredDecodeProgressSink
            retired_decode_progress_sink;
        if (config_.moe_rebalance.mode ==
                MoERebalanceRuntimeMode::Dynamic &&
            (moe_expert_overlay_maintenance_service_ ||
             moe_overlay_device_controller_graph_service_))
        {
            if (moe_expert_overlay_maintenance_service_ &&
                moe_overlay_device_controller_graph_service_)
            {
                return setError(
                    "Hosted ExpertOverlay decode progress has both host and device maintenance authorities");
            }
            const bool retain_terminal_ack =
                static_cast<bool>(moe_expert_overlay_maintenance_service_) ||
                local_rank == source_rank;
            retired_decode_progress_sink =
                [this, retain_terminal_ack](
                    std::uint64_t completed_tokens,
                    std::string *error) -> bool
            {
                if (error)
                    error->clear();
                if (completed_tokens == 0u ||
                    (static_cast<bool>(
                         moe_expert_overlay_maintenance_service_) ==
                     static_cast<bool>(
                         moe_overlay_device_controller_graph_service_)))
                {
                    if (error)
                    {
                        *error =
                            "Hosted ExpertOverlay decode progress reached an inactive maintenance authority";
                    }
                    return false;
                }

                if (moe_overlay_device_controller_graph_service_)
                {
                    if (!moe_overlay_device_controller_graph_service_
                             ->notifyInferenceProgress(
                                 MoEOverlayDeviceControllerGraphService::
                                     InferencePhase::Decode,
                                 completed_tokens))
                    {
                        if (error)
                        {
                            *error =
                                "Hosted ExpertOverlay decode progress could not wake the device controller";
                            // Preserve the background authority's first fault.
                            // A failed wake is only its terminal propagation
                            // boundary, not the failed transfer/publication edge.
                            const auto cause = moe_overlay_device_controller_graph_service_->failureMessage();
                            if (!cause.empty())
                                *error += ": " + cause;
                        }
                        return false;
                    }
                }
                else
                {
                    moe_expert_overlay_maintenance_service_
                        ->notifyInferenceProgress({
                            .phase =
                                MoEOverlayInferenceProgressPhase::Decode,
                            .completed_logical_tokens = completed_tokens,
                        });
                }

                if (!retain_terminal_ack)
                    return true;

                auto &retired =
                    hosted_device_generation_decode_progress_retired_;
                std::uint64_t observed = retired.load(
                    std::memory_order_relaxed);
                while (completed_tokens <=
                       std::numeric_limits<std::uint64_t>::max() - observed)
                {
                    if (retired.compare_exchange_weak(
                            observed,
                            observed + completed_tokens,
                            std::memory_order_release,
                            std::memory_order_relaxed))
                    {
                        return true;
                    }
                }
                if (error)
                {
                    *error =
                        "Hosted ExpertOverlay decode progress receipt overflowed";
                }
                return false;
            };
        }

        try
        {
            if (local_rank == source_rank)
            {
                std::vector<std::shared_ptr<
                    IMoEOverlayInferenceTransactionPublisher>> publishers;
                for (const auto &rank_plan : execution->rank_plans)
                {
                    if (!rank_plan.usesExpertTransactionFollower())
                    {
                        continue;
                    }
                    if (!rank_plan.loads_expert_weights ||
                        rank_plan.world_rank == source_rank)
                    {
                        return setError(
                            "ExpertOverlay execution planning produced an "
                            "invalid expert-only transaction follower");
                    }

                    const auto topology =
                        makeMoEOverlayInferenceTopologyIdentity(
                            snapshot->originalOwnerMap(),
                            graph_family,
                            source_rank,
                            rank_plan.world_rank);
                    auto channel = std::make_shared<
                        MoEOverlayMPIInferenceTransactionChannel>(
                        MoEOverlayMPIInferenceTransactionChannel::Config{
                            .mpi_ctx = overlay_context,
                            .source_world_rank = source_rank,
                            .target_world_rank = rank_plan.world_rank,
                            .send_slot_count = slot_count,
                        });
                    publishers.push_back(std::make_shared<
                        MoEOverlayInferenceTransactionPublisher>(
                        MoEOverlayInferenceTransactionPublisher::Config{
                            .channel = std::move(channel),
                            .protocol = {
                                .topology = topology,
                                .slot_count = slot_count,
                                .max_request_count = max_request_count,
                                .max_rows_per_request = max_graph_rows,
                                .max_mtp_draft_depth =
                                    max_mtp_draft_depth,
                            },
                        }));
                }
                if (local_endpoints)
                {
                    if (local_rank != source_rank)
                        return setError("Original same-rank experts have no continuation command authority");
                    publishers.push_back(MoEOverlayParticipantGraphRunner::createRankLocalInferencePublisher(
                        local_endpoints));
                }
                if (publishers.empty())
                {
                    return setError(
                        "ExpertOverlay transaction authority lost every "
                        "planner-declared expert-only follower");
                }

                int continuation_participants = 1;
                if (const auto *rank_runner =
                        dynamic_cast<const RankOrchestrator *>(runner_.get()))
                {
                    continuation_participants = rank_runner->device_count();
                }
                const int logical_root_participant =
                    snapshot->placement_plan->continuation_domain_spec
                        .logical_root_participant;
                const auto *const ticket_authority =
                    snapshot->originalOwnerMap().participantForId(
                        logical_root_participant);
                if (!ticket_authority ||
                    ticket_authority->domain_name !=
                        execution->continuation_domain ||
                    !ticket_authority->world_rank_known ||
                    ticket_authority->world_rank != source_rank ||
                    ticket_authority->domain_participant_index < 0 ||
                    ticket_authority->domain_participant_index >=
                        continuation_participants)
                {
                    return setError(
                        "Distributed ExpertOverlay could not map the "
                        "planner-declared continuation root to one local "
                        "ticket-authority graph participant");
                }

                std::vector<MoEOverlayInferenceCompletionBoundaryKind>
                    participant_completion_boundaries(
                        static_cast<std::size_t>(continuation_participants),
                        MoEOverlayInferenceCompletionBoundaryKind::Unspecified);
                for (const auto &participant :
                     snapshot->originalOwnerMap().participants())
                {
                    if (participant.domain_name !=
                            execution->continuation_domain ||
                        !participant.world_rank_known ||
                        participant.world_rank != source_rank)
                    {
                        continue;
                    }
                    if (participant.domain_participant_index < 0 ||
                        participant.domain_participant_index >=
                            continuation_participants)
                    {
                        return setError(
                            "Distributed ExpertOverlay continuation participant has an out-of-range LocalTP index");
                    }
                    const std::size_t participant_index =
                        static_cast<std::size_t>(
                            participant.domain_participant_index);
                    const auto resolved_boundary =
                        participant.device.is_gpu()
                            ? MoEOverlayInferenceCompletionBoundaryKind::
                                  DeviceEvent
                            : MoEOverlayInferenceCompletionBoundaryKind::
                                  HostSynchronous;
                    auto &completion_boundary =
                        participant_completion_boundaries[participant_index];
                    if (completion_boundary !=
                            MoEOverlayInferenceCompletionBoundaryKind::
                                Unspecified &&
                        completion_boundary != resolved_boundary)
                    {
                        return setError(
                            "Distributed ExpertOverlay continuation topology maps conflicting device kinds to one LocalTP participant index");
                    }
                    /* Repeated domain use across routed tiers is consistent. */
                    completion_boundary = resolved_boundary;
                }
                if (std::any_of(
                        participant_completion_boundaries.begin(),
                        participant_completion_boundaries.end(),
                        [](MoEOverlayInferenceCompletionBoundaryKind boundary)
                        {
                            return boundary ==
                                   MoEOverlayInferenceCompletionBoundaryKind::
                                       Unspecified;
                        }))
                {
                    return setError(
                        "Distributed ExpertOverlay could not resolve an exact completion boundary for every continuation participant");
                }

                /* The rank-synchronized serving-family phase completed before
                 * transaction authority installation. Freeze that production
                 * fact into one global plan now: one continuation-rank segment
                 * and one complete retained transaction per expert-follower
                 * rank. A remote rank with several GPUs remains one ticket
                 * boundary while retaining its exact local endpoint count. */
                const auto materialization_for_devices =
                    [](const OverlayRankPlan &rank_plan)
                {
                    if (rank_plan.local_devices.empty() ||
                        std::any_of(
                            rank_plan.local_devices.begin(),
                            rank_plan.local_devices.end(),
                            [](DeviceId device)
                            {
                                return !device.is_valid();
                            }))
                    {
                        throw std::invalid_argument(
                            "ExpertOverlay graph-plan rank owns no valid local participant devices");
                    }
                    const bool owns_gpu = std::any_of(
                        rank_plan.local_devices.begin(),
                        rank_plan.local_devices.end(),
                        [](DeviceId device)
                        {
                            return device.is_gpu();
                        });
                    return owns_gpu
                               ? MoEOverlayInferenceSegmentMaterializationKind::
                                     NativeDeviceExecutable
                               : MoEOverlayInferenceSegmentMaterializationKind::
                                     EagerHostGraph;
                };

                MoEOverlayInferenceSegmentMaterializationKind
                    continuation_materialization;
                switch (runner_->servingGraphPreparationKind())
                {
                case ServingGraphPreparationKind::
                    NativeDeviceExecutableFamily:
                    continuation_materialization =
                        MoEOverlayInferenceSegmentMaterializationKind::
                            NativeDeviceExecutable;
                    break;
                case ServingGraphPreparationKind::EagerHostGraph:
                    continuation_materialization =
                        MoEOverlayInferenceSegmentMaterializationKind::
                            EagerHostGraph;
                    break;
                case ServingGraphPreparationKind::Unresolved:
                    return setError(
                        "Distributed ExpertOverlay continuation graph has no sealed serving-family materialization kind");
                }

                std::vector<
                    MoEOverlayInferenceCoordinatorGraphPlan::Segment>
                    coordinator_segments;
                coordinator_segments.reserve(
                    1u + execution->rank_plans.size());
                coordinator_segments.push_back({
                    .role =
                        MoEOverlayInferenceCoordinatorSegmentRole::Continuation,
                    .materialization = continuation_materialization,
                    .world_rank = source_rank,
                    .local_participant_count =
                        static_cast<std::size_t>(continuation_participants),
                });
                if (local_endpoints)
                {
                    coordinator_segments.push_back({
                        .role = MoEOverlayInferenceCoordinatorSegmentRole::ExpertFollower,
                        .materialization = MoEOverlayInferenceSegmentMaterializationKind::NativeDeviceExecutable,
                        .world_rank = local_rank,
                        .local_participant_count = local_endpoints->participantIds().size(),
                    });
                }
                for (const auto &rank_plan : execution->rank_plans)
                {
                    if (!rank_plan.usesExpertTransactionFollower())
                        continue;
                    coordinator_segments.push_back({
                        .role = MoEOverlayInferenceCoordinatorSegmentRole::
                            ExpertFollower,
                        .materialization =
                            materialization_for_devices(rank_plan),
                        .world_rank = rank_plan.world_rank,
                        .local_participant_count =
                            rank_plan.local_devices.size(),
                    });
                }
                auto coordinator_graph_plan =
                    MoEOverlayInferenceCoordinatorGraphPlan::
                        sealAfterSynchronizedMaterialization(
                            graph_family.graph_family_generation,
                            std::move(coordinator_segments));
                auto coordinator = std::make_shared<
                    MoEOverlayInferenceTransactionCoordinator>(
                    MoEOverlayInferenceTransactionCoordinator::Config{
                        .publishers = std::move(publishers),
                        .graph_plan = std::move(coordinator_graph_plan),
                        .continuation_participant_count =
                            continuation_participants,
                        .ticket_authority_participant_index =
                            ticket_authority->domain_participant_index,
                        .participant_completion_boundaries =
                            std::move(participant_completion_boundaries),
                        .max_transactions_per_command = slot_count,
                        .max_mtp_draft_depth = max_mtp_draft_depth,
                        .retired_prefill_progress_sink =
                            retired_prefill_progress_sink,
                        .retired_decode_progress_sink =
                            retired_decode_progress_sink,
                        .residency_authority =
                            moe_expert_overlay_residency_authority_,
                    });
                if (config_.moe_rebalance.mode ==
                    MoERebalanceRuntimeMode::Dynamic)
                {
                    std::string probe_error;
                    if (!coordinator->bindPrefillInterferenceProbe(
                            moe_expert_overlay_interference_probe_,
                            &probe_error))
                    {
                        return setError(
                            "Distributed Dynamic ExpertOverlay could not bind "
                            "its aggregate prefill calibration authority: " +
                            probe_error);
                    }
                    moe_overlay_prefill_interference_owner_ =
                        MoEOverlayPrefillInterferenceOwner::
                            TransactionCoordinator;
                }
                if (!runner_->setMoEOverlayInferenceTransactionCoordinator(
                        coordinator,
                        /*continuation_participant_index=*/0))
                {
                    return setError(
                        "Continuation runner rejected the rank-wide "
                        "ExpertOverlay inference transaction authority");
                }
                moe_overlay_inference_transaction_coordinator_ =
                    std::move(coordinator);
            }
            else
            {
                const auto &rank_plan = execution->currentRankPlan();
                if (!rank_plan.usesExpertTransactionFollower())
                {
                    return setError(
                        "Only an expert-only rank may enter the retained-graph "
                        "transaction follower lifecycle");
                }
                if (!rank_plan.loads_expert_weights)
                {
                    return setError(
                        "Distributed ExpertOverlay worker rank owns neither a "
                        "continuation graph nor a retained expert graph");
                }
                auto *participant_runner =
                    dynamic_cast<MoEOverlayParticipantGraphRunner *>(
                        runner_.get());
                if (!participant_runner)
                {
                    return setError(
                        "Remote ExpertOverlay rank did not build the retained "
                        "participant graph runner required by its follower");
                }

                auto protocol =
                    participant_runner->inferenceTransactionProtocolConfig();
                if (protocol.topology.source_world_rank != source_rank ||
                    protocol.topology.target_world_rank != local_rank ||
                    protocol.slot_count != slot_count ||
                    protocol.max_request_count != max_request_count ||
                    protocol.max_rows_per_request != max_graph_rows ||
                    protocol.max_mtp_draft_depth != max_mtp_draft_depth)
                {
                    return setError(
                        "Remote ExpertOverlay retained graph family disagrees "
                        "with orchestration transaction geometry");
                }
                auto channel = std::make_shared<
                    MoEOverlayMPIInferenceTransactionChannel>(
                    MoEOverlayMPIInferenceTransactionChannel::Config{
                        .mpi_ctx = overlay_context,
                        .source_world_rank = source_rank,
                        .target_world_rank = local_rank,
                        .send_slot_count = slot_count,
                    });
                moe_overlay_inference_transaction_follower_ =
                    std::make_unique<
                        MoEOverlayInferenceTransactionFollower>(
                        MoEOverlayInferenceTransactionFollower::Config{
                            .channel = std::move(channel),
                            .executor = participant_runner,
                            .protocol = std::move(protocol),
                            .interference_probe =
                                moe_expert_overlay_interference_probe_,
                            .retired_prefill_progress_sink =
                                retired_prefill_progress_sink,
                            .retired_decode_progress_sink =
                                retired_decode_progress_sink,
                        });
            }
        }
        catch (const std::exception &error)
        {
            return setError(
                std::string(
                    "Failed to initialize ExpertOverlay inference "
                    "transactions: ") +
                error.what());
        }

        PerfStatsCollector::addCounter(
            "moe_overlay_transaction",
            "control_plane_initializations",
            1.0,
            "graph_setup",
            {},
            {{"execution_kind",
              toString(local_rank_plan.execution_kind)},
             {"control_plane",
              local_rank_plan.ownsTransactionAuthority()
                  ? "retained_graph_ticket_publisher"
                  : "retained_graph_ticket_follower"},
             {"source_rank", std::to_string(source_rank)},
             {"local_rank", std::to_string(local_rank)},
             {"expert_followers",
              std::to_string(expert_follower_count)},
             {"slots", std::to_string(slot_count)},
             {"max_mtp_depth", std::to_string(max_mtp_draft_depth)}});
        return true;
    }

} // namespace llaminar2
