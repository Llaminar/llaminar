/**
 * @file MoEOverlayParticipantGraphRunner.cpp
 * @brief Participant graph runner setup, graph variant construction, and runtime lifecycle.
 */
#include "MoEOverlayParticipantGraphRunnerInternal.h"

namespace llaminar2
{
    void requireMoEOverlayParticipantGPUOwnership(
        const MoERoutedExpertPlacementPlan &plan,
        const MoEExpertOwnerMap &original_owners,
        int world_rank,
        MoEOverlayParticipantGraphOwnership ownership,
        std::span<const int> fabric_local_ids,
        std::span<const int> follower_gpu_ids)
    {
        if (world_rank < 0 || plan.continuation_domain.empty() ||
            (ownership != MoEOverlayParticipantGraphOwnership::ExpertOnlyRank &&
             ownership != MoEOverlayParticipantGraphOwnership::RankLocalAuxiliary))
        {
            throw std::invalid_argument(
                "Mapped follower GPU ownership has an invalid original rank or role");
        }

        // Both expectations come from the same original catalog. The fabric
        // spans the rank; continuation GPU execution remains solely DGO-owned.
        std::vector<int> expected_fabric;
        std::vector<int> expected_followers;
        for (const auto &participant : original_owners.participants())
        {
            if (!participant.device.is_gpu())
                continue;
            if (!participant.world_rank_known || participant.world_rank < 0)
                throw std::invalid_argument(
                    "Mapped follower GPU ownership requires original GPU rank identity");
            if (participant.world_rank != world_rank)
                continue;
            expected_fabric.push_back(participant.participant_id);
            if (participant.domain_name != plan.continuation_domain)
                expected_followers.push_back(participant.participant_id);
            else if (ownership == MoEOverlayParticipantGraphOwnership::ExpertOnlyRank)
                throw std::invalid_argument(
                    "Expert-only follower rank cannot own a continuation GPU endpoint");
        }

        // Equality after duplicate rejection permits arbitrary catalog order,
        // but never treats an arbitrary subset as a valid ownership projection.
        const auto sorted_unique = [](std::span<const int> ids)
        {
            std::vector<int> result(ids.begin(), ids.end());
            std::sort(result.begin(), result.end());
            if ((!result.empty() && result.front() < 0) ||
                std::adjacent_find(result.begin(), result.end()) != result.end())
                throw std::invalid_argument(
                    "Mapped follower GPU ownership has a negative or duplicate participant ID");
            return result;
        };
        if (sorted_unique(fabric_local_ids) != sorted_unique(expected_fabric))
            throw std::invalid_argument(
                "Mapped controller-fabric membership differs from original local GPU owners");
        if (sorted_unique(follower_gpu_ids) != sorted_unique(expected_followers))
            throw std::invalid_argument(
                "Mapped follower GPU membership differs from original non-continuation GPU owners");
    }

    MoEOverlayParticipantGraphRunner::MoEOverlayParticipantGraphRunner(
        MoEOverlayParticipantGraphRunnerConfig config)
        : config_(std::move(config))
    {
        if (config_.ownership != MoEOverlayParticipantGraphOwnership::ExpertOnlyRank &&
            config_.ownership != MoEOverlayParticipantGraphOwnership::RankLocalAuxiliary)
            throw std::invalid_argument("Participant graph has an invalid original ownership scope");
        if (!config_.model_context || !config_.mpi_context ||
            !config_.placement_plan)
        {
            throw std::invalid_argument(
                "MoE overlay participant runner requires model, MPI, and "
                "placement authorities");
        }
        if (!config_.placement_plan->usesExpertOverlayAuthority())
        {
            throw std::invalid_argument(
                "MoE overlay participant runner requires a tiered overlay plan");
        }
        if (config_.max_seq_len <= 0)
        {
            throw std::invalid_argument(
                "MoE overlay participant runner requires a positive maximum "
                "sequence length");
        }
        if (config_.max_graph_activation_rows <= 0)
        {
            throw std::invalid_argument(
                "MoE overlay participant runner requires a positive planner-admitted "
                "graph activation capacity");
        }
        const auto normalized_prefill_shapes =
            normalizePrefillGraphBuckets(config_.prefill_graph_row_shapes);
        if (normalized_prefill_shapes != config_.prefill_graph_row_shapes ||
            normalized_prefill_shapes.empty() ||
            normalized_prefill_shapes.back() >
                config_.max_graph_activation_rows)
        {
            throw std::invalid_argument(
                "MoE overlay participant runner requires the exact sorted "
                "prefill-row manifest admitted by the distributed schedule");
        }
        if (config_.max_decode_activation_rows <= 0 ||
            config_.max_decode_activation_rows >
                config_.max_graph_activation_rows)
        {
            throw std::invalid_argument(
                "MoE overlay participant runner decode/MTP row capacity must "
                "be positive and fit the planner-admitted graph capacity");
        }
        const bool retains_mtp_graph_family =
            config_.mtp_graph_family_policy ==
            MoEOverlayMTPGraphFamilyPolicy::RetainModelSidecars;
        if (retains_mtp_graph_family &&
            config_.max_mtp_draft_depth <= 0)
        {
            throw std::invalid_argument(
                "MTP-capable participant runner requires a positive admitted draft depth");
        }
        if (!retains_mtp_graph_family &&
            config_.max_mtp_draft_depth != 0)
        {
            throw std::invalid_argument(
                "Main-only participant runner cannot advertise an MTP draft capacity");
        }
        if (config_.max_request_count <= 0)
        {
            throw std::invalid_argument(
                "Participant runner requires a positive transaction request capacity");
        }

        architecture_ = config_.model_context->architecture();
        residency_authority_ = config_.residency_authority;
        decode_histogram_ = config_.decode_histogram;
        if (!residency_authority_ || !config_.participant_residency)
        {
            throw std::invalid_argument(
                "Production MoE overlay participant runner requires the shared "
                "residency authority and process-local prepared-bank registry");
        }
        {
            ScopedWeightLoadDetailTimer timer(
                "overlay.graph.resolve_topology");
            resolveTopology();
        }
        {
            ScopedWeightLoadDetailTimer timer(
                "overlay.graph.resolve_families");
            resolveGraphFamilies();
        }
        if (config_.ownership == MoEOverlayParticipantGraphOwnership::RankLocalAuxiliary)
        {
            // The exact roster must exist before any graph/event creator. The
            // later publisher borrows this SAME state, never another protocol.
            rank_local_publication_state_ = std::make_unique<RankLocalPublicationState>(
                inferenceTransactionProtocolConfig());
            if (!config_.rank_batch_transport_registry)
                throw std::logic_error("Original rank-local protocol has no cold mapped transport registry");
            original_native_slot_geometry_ = config_.rank_batch_transport_registry->originalNativeSlotGeometryFor(
                transaction_topology_identity_.source_world_rank,
                transaction_topology_identity_.target_world_rank,
                admitted_rank_local_endpoint_ids_);
            if (original_native_slot_geometry_)
            {
                const auto domain = rank_local_publication_state_->originalSlotDomain();
                if (*original_native_slot_geometry_ != domain->geometry())
                    throw std::logic_error("Original native channel geometry differs from its sole protocol roster");
                config_.rank_batch_transport_registry->bindOriginalNativeSlotsFor(domain,
                    transaction_topology_identity_.source_world_rank,
                    transaction_topology_identity_.target_world_rank,
                    admitted_rank_local_endpoint_ids_);
            }
        }
        {
            /*
             * Every native follower graph on one GPU draws from the same
             * opaque CUDA/HIP pool. Commit the complete admitted family before
             * creating contexts or graph owners; an instantiation-time memory
             * delta is evidence about this pool and cannot become a second
             * per-executable accounting authority.
             */
            const auto weight_manager =
                config_.model_context->concreteWeightManager();
            const auto memory_authority = weight_manager
                                              ? weight_manager
                                                    ->physicalMemoryAuthority()
                                              : nullptr;
            if (!memory_authority)
            {
                throw std::logic_error(
                    "Participant graph has no canonical physical-memory authority for native graph-pool admission");
            }
            for (const DeviceId device :
                 participantDevices(local_participants_))
            {
                if (!device.is_gpu())
                    continue;
                if (config_.ownership == MoEOverlayParticipantGraphOwnership::RankLocalAuxiliary &&
                    std::none_of(local_participants_.begin(), local_participants_.end(),
                        [&](const auto *participant)
                        {
                            return participant->device == device &&
                                std::binary_search(admitted_rank_local_endpoint_ids_.begin(),
                                                   admitted_rank_local_endpoint_ids_.end(),
                                                   participant->participant_id);
                        }))
                    continue;
                const std::size_t admitted_graph_bytes =
                    memory_authority->plannedBytes(
                        device,
                        PhysicalMemoryOwner::NativeGraphExecutable);
                if (admitted_graph_bytes == 0u)
                {
                    throw std::logic_error(
                        "Participant GPU graph family has zero native graph-pool admission for " +
                        device.toString());
                }
                native_graph_memory_reservations_.emplace(
                    device,
                    std::make_shared<PhysicalMemoryOwnerReservation>(
                        memory_authority->reserveNewAllocations(
                            device,
                            PhysicalMemoryOwner::NativeGraphExecutable,
                            admitted_graph_bytes)));
            }
            serial_graph_family_workspace_allocator_ =
                std::make_shared<WorkspaceAllocator>(memory_authority);
        }
        {
            ScopedWeightLoadDetailTimer timer(
                "overlay.graph.create_compact_arenas");
            createSerialCompactBufferArenas();
        }
        {
            ScopedWeightLoadDetailTimer timer(
                "overlay.graph.prepare_participant_weights");
            prepareParticipantWeights();
        }
        // Participant weights are complete before any expert graph is declared.
        // Establish original context/source identity here, not from a lazy builder.
        {
            std::string error;
            if (!config_.participant_residency->admitInitialJointGraphSources(
                    config_.model_context, &error))
                throw std::runtime_error("Participant CPU joint cold graph admission failed: " + error);
        }
        {
            ScopedWeightLoadDetailTimer timer(
                "overlay.graph.create_device_contexts");
            createDeviceContexts();
        }
        {
            ScopedWeightLoadDetailTimer timer(
                "overlay.graph.create_participant_gpu_runtimes");
            createParticipantGpuRuntimes();
        }
        /*
         * The residency-maintenance service starts immediately after every
         * rank finishes runner construction. Materialize the complete bounded
         * graph now so each local stage resolves its exact prepared engines
         * and atomically publishes the initial participant bank before that
         * service can inspect or clone an epoch. The sparse packet supplies
         * the live row count later; graph addresses and capacity never change.
         */
        ParticipantGraphBuildSpec main_spec;
        main_spec.role = ParticipantGraphFamilyRole::Main;
        main_spec.row_capacity = config_.max_graph_activation_rows;
        main_spec.source_layers.reserve(
            static_cast<size_t>(main_layer_count_));
        for (int row = 0; row < main_layer_count_; ++row)
            main_spec.source_layers.push_back(transaction_graph_family_.first_model_layer + row);
        {
            ScopedWeightLoadDetailTimer timer(
                "overlay.graph.build_main_family");
            main_graph_ = buildGraph(main_spec);
        }
        if (!main_graph_)
        {
            throw std::runtime_error(
                "MoE overlay participant main graph construction returned null");
        }

        {
            ScopedWeightLoadDetailTimer timer(
                "overlay.graph.build_mtp_families");
            mtp_sidecar_graphs_.reserve(mtp_source_layers_.size());
            for (size_t depth = 0; depth < mtp_source_layers_.size(); ++depth)
            {
                ParticipantGraphBuildSpec mtp_spec;
                mtp_spec.role = ParticipantGraphFamilyRole::MTPDraft;
                mtp_spec.source_layers = {mtp_source_layers_[depth]};
                mtp_spec.mtp_graph_depth = static_cast<int>(depth);
                mtp_spec.row_capacity = config_.max_decode_activation_rows;
                auto graph = buildGraph(mtp_spec);
                if (!graph)
                {
                    throw std::runtime_error(
                        "MoE overlay participant MTP sidecar graph construction returned null for graph depth " +
                        std::to_string(depth));
                }
                mtp_sidecar_graphs_.push_back(std::move(graph));
            }
        }
        for (const auto &[participant_id, runtime] :
             participant_gpu_runtimes_)
        {
            if (!runtime || !runtime->runtime_table ||
                runtime->initialized_layers.size() !=
                    static_cast<std::size_t>(
                        runtime->runtime_table->layerCount()) ||
                std::any_of(
                    runtime->initialized_layers.begin(),
                    runtime->initialized_layers.end(),
                    [](std::uint8_t initialized)
                    {
                        return initialized == 0u;
                    }))
            {
                throw std::runtime_error(
                    "Participant retained graph families did not publish every topology-wide device runtime layer for participant " +
                    std::to_string(participant_id));
            }
        }
        {
            // Continuation construction may defer its graph declarations until
            // this auxiliary owner is available. Prepared weights are already
            // complete on every local participant, including retained dormant
            // MTP layers. Publish those exact registry-owned banks before the
            // whole-cohort readiness check; building a graph is not a required
            // side effect of physical residency admission.
            const auto manager = config_.model_context->concreteWeightManager();
            std::string error;
            if (!manager ||
                !config_.participant_residency->bindOriginalPreparedContext(
                    config_.model_context,
                    config_.participant_residency->projectionPreparation(),
                    &error) ||
                !config_.participant_residency->finalizeInitialBanksFromPreparedRegistry(
                    manager->expertGemmRegistry(), &error))
            {
                throw std::runtime_error(
                    "Participant retained graph family initial prepared residency finalization failed: " +
                    (error.empty() ? std::string("missing original prepared engine registry") : error));
            }
        }
        if (!config_.participant_residency->allInitialBanksReady())
        {
            std::string deficits;
            for (const auto &bank :
                 config_.participant_residency->incompleteInitialBanks())
            {
                deficits += " participant=" + std::to_string(bank.participant_id) +
                    " device=" + bank.device.toString() + " missing_layers=";
                for (const int layer : bank.missing_layers)
                    deficits += std::to_string(layer) + ",";
                if (bank.publication_pending)
                    deficits += "publication_pending";
            }
            throw std::runtime_error(
                "Participant retained graph families completed construction before every local initial prepared residency bank became ready:" +
                deficits);
        }
        if(config_.model_context)config_.model_context->sealPreparedFollowerCaptureTerminals();
        releasePreparedSourceBytes();

        PerfStatsCollector::addCounter(
            "moe_overlay_participant_graph",
            "prepared_ranks",
            1.0,
            "setup",
            participantDeviceList(local_participants_),
            {
                {"participants", participantIdList(local_participants_)},
                {"world_rank", std::to_string(config_.mpi_context->rank())},
                {"weight_bindings",
                 std::to_string(
                     frozen_weights_ ? frozen_weights_->bindings().size() : 0u)},
                {"main_layers", std::to_string(main_layer_count_)},
                {"mtp_sidecar_graphs",
                 std::to_string(mtp_sidecar_graphs_.size())},
            });
    }

    MoEOverlayParticipantGraphRunner::~MoEOverlayParticipantGraphRunner() =
        default;

    ServingGraphPreparationKind
    MoEOverlayParticipantGraphRunner::servingGraphPreparationKind()
        const noexcept
    {
        if (local_participants_.empty() ||
            (config_.ownership == MoEOverlayParticipantGraphOwnership::RankLocalAuxiliary &&
             admitted_rank_local_endpoint_ids_.empty()))
            return ServingGraphPreparationKind::Unresolved;

        bool owns_gpu_endpoint = false;
        for (const auto *participant : local_participants_)
        {
            if (!participant || !participant->device.is_valid())
                return ServingGraphPreparationKind::Unresolved;
            owns_gpu_endpoint =
                owns_gpu_endpoint || participant->device.is_gpu();
        }

        /* CPU endpoints are already materialized in their retained eager
         * graph. A mixed follower still reports the stronger native-device
         * transition because every GPU endpoint must be sealed before the
         * rank can accept its first authenticated transaction. */
        return owns_gpu_endpoint
                   ? ServingGraphPreparationKind::
                         NativeDeviceExecutableFamily
                   : ServingGraphPreparationKind::EagerHostGraph;
    }

    bool MoEOverlayParticipantGraphRunner::
        installMoEOverlayTransferProgressEpoch(
            std::shared_ptr<MappedTransferProgressEpoch> epoch)
    {
        if (!epoch || !epoch->device().is_gpu() ||
            serving_graph_family_lifecycle_ !=
                ServingGraphFamilyLifecycle::Built)
        {
            LOG_ERROR(
                "[MoEOverlayParticipantGraphRunner] Transfer-progress epoch installation requires one local GPU before follower graph sealing");
            return false;
        }
        const DeviceId device = epoch->device();
        const bool owns_device = std::any_of(
            local_participants_.begin(),
            local_participants_.end(),
            [device](const auto *participant)
            {
                return participant && participant->device == device;
            });
        if (!owns_device)
        {
            LOG_ERROR(
                "[MoEOverlayParticipantGraphRunner] Transfer-progress epoch names a non-local follower device "
                << device.toString());
            return false;
        }

        const auto found = transfer_progress_epochs_.find(device);
        if (found != transfer_progress_epochs_.end() &&
            found->second != epoch)
        {
            LOG_ERROR(
                "[MoEOverlayParticipantGraphRunner] Refused to replace the transfer-progress authority on "
                << device.toString());
            return false;
        }
        transfer_progress_epochs_[device] = epoch;

        /* Every follower GPU owns one execution-state runtime. Transfer
         * progress is independent of whether host or device policy authors the
         * next placement, so bind the retained relay epoch to the same exact
         * inference-boundary owner in both regimes. */
        for (auto &[participant_id, runtime] : participant_gpu_runtimes_)
        {
            (void)participant_id;
            if (runtime && runtime->device == device &&
                !runtime->installMoEOverlayTransferProgressEpoch(epoch))
            {
                LOG_ERROR(
                    "[MoEOverlayParticipantGraphRunner] Could not bind transfer progress to a device-controller runtime on "
                    << device.toString());
                return false;
            }
        }
        return true;
    }

    void MoEOverlayParticipantGraphRunner::bindOriginalPooledSourceFamily(
        std::shared_ptr<qwen4exp::PooledImmutableRetainedParticipants> original,
        std::vector<RetainedForwardSourceDefinition> definitions)
    {
        if (!original || definitions.empty() || config_.ownership != MoEOverlayParticipantGraphOwnership::RankLocalAuxiliary)
            throw std::invalid_argument("Expert storage family requires its actual original Rank source");
        original->requireCurrent();
        for (const auto& definition : definitions) definition.requireCurrent();
        if (original_pooled_participants_) {
            if (original_pooled_participants_ != original ||
                original_pooled_participants_.owner_before(original) || original.owner_before(original_pooled_participants_))
                throw std::invalid_argument("Expert storage family changed its original source owner");
            const bool same = original_pooled_source_family_.size() == definitions.size() &&
                std::all_of(original_pooled_source_family_.begin(),original_pooled_source_family_.end(),[&](const auto& prior) {
                    return std::any_of(definitions.begin(),definitions.end(),[&](const auto& current) {
                        return prior.graph_identity == current.graph_identity && prior.source == current.source &&
                            !prior.source.owner_before(current.source) && !current.source.owner_before(prior.source) &&
                            prior.signature == current.signature && prior.role == current.role && prior.physical_rows == current.physical_rows &&
                            prior.topology_generation == current.topology_generation && prior.capture_variant_signature == current.capture_variant_signature;
                    });
                });
            if (same) return;
            requireColdSourceAliasRefresh(serving_graph_family_lifecycle_);
            if (serving_graph_family_lifecycle_ == ServingGraphFamilyLifecycle::Sealed) {
                std::vector<qwen4exp::PooledImmutableRetainedParticipants::OriginalExpertSourceBinding> bindings;
                const auto stage = [&](const CachedParticipantGraph* family) {
                    if (!family) return;
                    for (const auto& shape : family->mapped_gpu_shapes) for (const auto& endpoint : shape.endpoints) {
                        if (!endpoint || !original->hasOriginalExpertOwner(endpoint->device)) continue;
                        bindings.push_back({endpoint->device,endpoint->graph.get(),retainedExpertSourceVariants(definitions,
                            family->role == ParticipantGraphFamilyRole::MTPDraft,shape.physical_rows)});
                    }
                };
                stage(main_graph_.get());
                for (const auto& sidecar : mtp_sidecar_graphs_) stage(sidecar.get());
                original->refreshOriginalExpertSourceVariants(bindings);
            }
        }
        original_pooled_source_family_ = std::move(definitions);
        original_pooled_participants_ = std::move(original);
    }

    void MoEOverlayParticipantGraphRunner::cancelOriginalPooledSource(const std::function<void()>& cancellation,
        std::mutex& failure_mutex,std::exception_ptr& failure) noexcept
    {
        if (!cancellation) return;
        try { cancellation(); }
        catch (...) {
            // Invoke outside this metadata lock: the original service/channel
            // authority owns its own short mapped-publication locks and never
            // waits for an expert worker or native terminal here.
            std::lock_guard lock(failure_mutex);
            if (!failure) failure = std::current_exception();
        }
    }

    void MoEOverlayParticipantGraphRunner::bindOriginalPooledSourceCancellation(std::function<void()> cancellation)
    {
        if (!original_pooled_participants_) return;
        if (!cancellation)
            throw std::invalid_argument("Expert storage execution requires its original Rank source cancellation");
        // Rank replaces this only when its prior source invocation has crossed
        // the existing complete native/channel retirement frontier. Enqueued
        // original worker jobs never observe a concurrent request rebind.
        original_pooled_source_cancellation_ = std::make_shared<const std::function<void()>>(std::move(cancellation));
    }

    void MoEOverlayParticipantGraphRunner::retireOriginalPooledStorageAttachments()
    {
        if (!original_pooled_participants_) return;
        const auto retire = [&](CachedParticipantGraph* family) {
            if (!family) return;
            for (auto& shape : family->mapped_gpu_shapes) for (auto& endpoint : shape.endpoints) {
                if (!endpoint || !original_pooled_participants_->hasOriginalExpertOwner(endpoint->device)) continue;
                auto* worker = participantWorkerContext(endpoint->device);
                if (!worker || worker->ownsCurrentThread())
                    throw std::logic_error("Expert storage retirement requires its original off-worker owner");
                worker->submitAndWait([&] {
                    endpoint->cache.reset(DeviceGraphExecutor::GraphSegmentCache::StreamResetPolicy::Preserve);
                });
            }
        };
        retire(main_graph_.get());
        for (auto& sidecar : mtp_sidecar_graphs_) retire(sidecar.get());
        original_pooled_source_family_.clear();
        // Every original worker FIFO and native parent has drained above. The
        // callback owns no runner/model back-reference and dies only now.
        original_pooled_source_cancellation_ = {};
        original_pooled_participants_.reset();
    }

    bool MoEOverlayParticipantGraphRunner::
        materializeServingGraphFamilyWithoutLaunch(
            const ServingGraphFamilyMaterializationPlan &plan)
    {
        ScopedWeightLoadDetailTimer materialization_timer(
            "overlay.graph.materialize_serving_family");
        const auto normalized_buckets =
            normalizePrefillGraphBuckets(plan.prefill_bucket_rows);
        if (!plan.valid() || normalized_buckets != plan.prefill_bucket_rows ||
            normalized_buckets != config_.prefill_graph_row_shapes ||
            normalized_buckets.back() > config_.max_graph_activation_rows)
        {
            LOG_ERROR(
                "[MoEOverlayParticipantGraphRunner] Follower serving graph setup disagrees with the frozen distributed bucket inventory");
            return false;
        }

        const auto branch_factory_for = [this](DeviceId device)
        {
            const auto found = transfer_progress_epochs_.find(device);
            return found == transfer_progress_epochs_.end()
                ? GraphCaptureAuxiliaryBranchFactory{} : found->second->graphBranchFactory();
        };
        const auto endpoint_matches_final_identity =
            [&](const CachedParticipantGraph::MappedGPUFollowerEndpoint &endpoint)
        {
            const auto factory = branch_factory_for(endpoint.device);
            const bool owns_branch =
                static_cast<bool>(endpoint.cache.auxiliary_branch);
            return endpoint.cache.initialized &&
                   endpoint.cache.retained_full_graph_replay.valid() &&
                   endpoint.cache.executable_submission_state ==
                       DeviceGraphExecutor::GraphSegmentCache::
                           ExecutableSubmissionState::MaterializedUnlaunched &&
                   owns_branch == factory.valid() &&
                   (!factory.valid() ||
                    endpoint.cache.auxiliary_branch_authority ==
                        factory.authority_identity);
        };

        std::vector<CachedParticipantGraph *> families;
        if (main_graph_)
            families.push_back(main_graph_.get());
        for (auto &sidecar : mtp_sidecar_graphs_)
        {
            if (sidecar)
                families.push_back(sidecar.get());
        }
        if (families.empty())
        {
            LOG_ERROR(
                "[MoEOverlayParticipantGraphRunner] Follower serving setup has no declared graph family");
            return false;
        }

        const auto require_original_source_associations = [&] {
            if (!original_pooled_participants_) return;
            for (const auto& definition : original_pooled_source_family_)
                if (definition.source) original_pooled_participants_->requireInvocation(*definition.source);
        };

        if (serving_graph_family_lifecycle_ !=
            ServingGraphFamilyLifecycle::Built)
        {
            for (const CachedParticipantGraph *family : families)
            {
                if (!family)
                    return false;
                for (const auto &shape : family->mapped_gpu_shapes)
                {
                    for (const auto &endpoint : shape.endpoints)
                    {
                        if (!endpoint ||
                            !endpoint_matches_final_identity(*endpoint))
                        {
                            LOG_ERROR(
                                "[MoEOverlayParticipantGraphRunner] Repeated follower serving setup found stale graph-branch identity");
                            return false;
                        }
                    }
                }
            }
            require_original_source_associations();
            return true;
        }

        std::size_t total_native_transactions = 0u;
        std::size_t total_branch_transactions = 0u;
        try
        {
            for (CachedParticipantGraph *family : families)
            {
                if (!family)
                    throw std::logic_error(
                        "Follower serving family retained a null graph owner");
                if (!family->usesMappedActivationEpochs())
                    continue;

                const std::size_t graph_family_ordinal =
                    family->role == ParticipantGraphFamilyRole::Main
                        ? 0u
                        : static_cast<std::size_t>(
                              family->mtp_graph_depth + 1);
                std::size_t family_native_transactions = 0u;
                std::size_t family_branch_transactions = 0u;

                for (auto &shape : family->mapped_gpu_shapes)
                {
                    for (auto &endpoint : shape.endpoints)
                    {
                        if (!endpoint || !endpoint->graph ||
                            !endpoint->executor ||
                            !endpoint->cache.capture_stream ||
                            endpoint->cache.initialized)
                        {
                            throw std::runtime_error(
                                "Mapped follower endpoint is not pristine at final serving-graph materialization");
                        }
                        const auto context_it =
                            execution_contexts_.find(endpoint->device);
                        IWorkerGPUContext *const worker =
                            participantWorkerContext(endpoint->device);
                        if (context_it == execution_contexts_.end() ||
                            !context_it->second || !worker ||
                            endpoint->graph->nativeCaptureEnvelope() !=
                                GraphNativeCaptureEnvelope::
                                    DeviceOwnedTimelineTransaction)
                        {
                            throw std::runtime_error(
                                "Mapped follower could not resolve its final native capture owner");
                        }

                        auto branch_factory = branch_factory_for(endpoint->device);
                        if (original_pooled_participants_ && original_pooled_participants_->hasOriginalExpertOwner(endpoint->device)) {
                            const auto original = original_pooled_participants_->originalParticipant(endpoint->device);
                            const auto actual = GPUDeviceContextPool::instance().retainExistingPreparedContext(endpoint->device,original.memory());
                            original.requireWorker(actual.worker());
                            if (actual.worker().get() != worker || actual.generation() != original.generation())
                                throw std::invalid_argument("Expert storage parent changed its original worker/PMA generation");
                            auto sources = retainedExpertSourceVariants(original_pooled_source_family_,
                                family->role == ParticipantGraphFamilyRole::MTPDraft,shape.physical_rows);
                            if (!sources.empty()) {
                                original_pooled_participants_->bindOriginalExpertSourceVariants(endpoint->device,*endpoint->graph,sources);
                                const auto source = sources.front();
                                branch_factory.source_definition_boundary = [source](const std::string&,void*,LocalTPCaptureSourceDefinition& handoff) {
                                    source->requireCurrentDefinition();
                                    if (handoff.publication || handoff.domain != source->definition().domain)
                                        throw std::invalid_argument("Expert storage capture changed its original source-family definition");
                                    handoff.result = source;
                                    return true;
                                };
                            }
                        }
                        bool materialized = false;
                        worker->submitAndWait(
                            [&]()
                            {
                                endpoint->graph->reset();
                                materialized = endpoint->executor
                                                   ->executeWithCachedGraphReplay(
                                                       *endpoint->graph,
                                                       context_it->second,
                                                       endpoint->cache,
                                                       endpoint->cache.capture_stream,
                                                       worker,
                                                       /*collective_nodes=*/nullptr,
                                                       /*collectives_graph_capturable=*/false,
                                                       /*force_recapture=*/false,
                                                       /*defer_final_sync=*/true,
                                                       {},
                                                       DeviceGraphExecutor::
                                                           GraphReplayPlanPolicy::
                                                               RequireFullGraph,
                                                       {},
                                                       {},
                                                       {},
                                                       DeviceGraphExecutor::
                                                           GraphInitialSubmissionPolicy::
                                                               MaterializeWithoutLaunch,
                                                       branch_factory) &&
                                               endpoint->cache
                                                   .prepareCaptureStreamTerminal(
                                                       worker);
                            });
                        if (!materialized ||
                            !endpoint_matches_final_identity(*endpoint))
                        {
                            throw std::runtime_error(
                                "Mapped follower could not seal its graph-owned transfer branch for participant " +
                                std::to_string(endpoint->participant_id) +
                                " rows=" +
                                std::to_string(shape.physical_rows));
                        }
                        ++family_native_transactions;
                        if (branch_factory.valid())
                            ++family_branch_transactions;
                    }
                }

                /* Every row shape for one participant borrows the same stream.
                 * One exact event fence per device proves setup left no work in
                 * front of transaction zero; it is never an inference wait. */
                std::vector<DeviceId> quiescent_devices;
                for (auto &shape : family->mapped_gpu_shapes)
                {
                    for (auto &endpoint : shape.endpoints)
                    {
                        if (!endpoint ||
                            std::find(
                                quiescent_devices.begin(),
                                quiescent_devices.end(),
                                endpoint->device) != quiescent_devices.end())
                        {
                            continue;
                        }
                        IWorkerGPUContext *const worker =
                            participantWorkerContext(endpoint->device);
                        if (!worker)
                        {
                            throw std::runtime_error(
                                "Mapped follower setup lost its participant worker context");
                        }
                        worker->submitAndWait(
                            [&]()
                            {
                                endpoint->cache.waitForCaptureStreamFence(
                                    DeviceGraphExecutor::GraphSegmentCache::
                                        HostFenceWaitPolicy::ActiveProgress,
                                    [device = endpoint->device,
                                     family_ordinal = graph_family_ordinal,
                                     rows = endpoint->physical_rows]()
                                    {
                                        return
                                            "final mapped follower graph materialization left queued work on device=" +
                                            device.toString() +
                                            " graph_family=" +
                                            std::to_string(family_ordinal) +
                                            " rows=" +
                                            std::to_string(rows);
                                    });
                            });
                        quiescent_devices.push_back(endpoint->device);
                    }
                }

                PerfStatsCollector::addCounter(
                    "moe_overlay_participant_graph",
                    "mapped_follower_setup_stream_quiescence_proofs",
                    static_cast<double>(quiescent_devices.size()),
                    "model_setup",
                    participantDeviceList(local_participants_),
                    {{"graph_family", std::to_string(graph_family_ordinal)},
                     {"ordering", "exact_stream_event"},
                     {"host_blocking", "setup_only"}});
                PerfStatsCollector::addCounter(
                    "moe_overlay_participant_graph",
                    "materialized_mapped_follower_families",
                    1.0,
                    "model_setup",
                    participantDeviceList(local_participants_),
                    {{"graph_family", std::to_string(graph_family_ordinal)},
                     {"row_capacity", std::to_string(family->row_capacity)},
                     {"setup_materialized_gpu_transactions",
                      std::to_string(family_native_transactions)},
                     {"setup_materialized_cpu_endpoints",
                      std::to_string(
                          family->mapped_cpu_endpoints.size())},
                     {"captured_transfer_branches",
                      std::to_string(family_branch_transactions)},
                     {"standalone_progress_launch", "false"}});
                total_native_transactions += family_native_transactions;
                total_branch_transactions += family_branch_transactions;
            }
        }
        catch (const std::exception &error)
        {
            LOG_ERROR(
                "[MoEOverlayParticipantGraphRunner] Final follower graph materialization failed: "
                << error.what());
            return false;
        }
        catch (...)
        {
            LOG_ERROR(
                "[MoEOverlayParticipantGraphRunner] Final follower graph materialization threw a non-standard exception");
            return false;
        }

        require_original_source_associations();
        serving_graph_family_lifecycle_ =
            ServingGraphFamilyLifecycle::Sealed;
        PerfStatsCollector::addCounter(
            "moe_overlay_participant_graph",
            "mapped_follower_serving_graph_family_completions",
            1.0,
            "model_setup",
            participantDeviceList(local_participants_),
            {{"native_transactions",
              std::to_string(total_native_transactions)},
             {"captured_transfer_branches",
              std::to_string(total_branch_transactions)},
             {"authority_installation", "before_native_capture"}});
        return true;
    }

    /**
     * @brief Resolve immutable rank-local endpoint ownership from the topology.
     *
     * The runtime and execution plans are both derived from the supplied
     * placement authority and MPI rank.  The resulting raw participant
     * pointers refer to @ref owner_map_, whose lifetime exceeds every cached
     * graph, so graph construction never re-resolves topology from a mutable
     * request path.
     */
    void MoEOverlayParticipantGraphRunner::resolveTopology()
    {
        runtime_plan_ = resolveMoEExpertOverlayRuntimePlan(
            config_.placement_plan,
            MoEExpertOverlayRuntimeResolverOptions{
                .current_world_rank = config_.mpi_context->rank(),
                .validate_mvp_root_reachability = false,
            });
        if (!runtime_plan_)
            throw std::runtime_error("Failed to resolve MoE overlay runtime plan");

        execution_plan_ =
            std::make_shared<MoEExpertOverlayExecutionPlan>(
                buildMoEExpertOverlayExecutionPlan(
                    *runtime_plan_, config_.mpi_context->world_size()));
        const auto &rank_plan = execution_plan_->currentRankPlan();
        const bool owns_original_protocol =
            config_.ownership == MoEOverlayParticipantGraphOwnership::ExpertOnlyRank
                ? rank_plan.usesExpertTransactionFollower()
                : rank_plan.ownsTransactionAuthority();
        if (!owns_original_protocol)
        {
            throw std::invalid_argument(
                "Retained expert ownership disagrees with its original "
                "rank transaction role; resolved execution kind=" +
                std::string(toString(
                    execution_plan_->currentRankPlan().execution_kind)));
        }

        const auto snapshot = residency_authority_->snapshot();
        if (!snapshot || !snapshot->valid() || !snapshot->placement_plan)
        {
            throw std::invalid_argument(
                "MoE overlay participant runner received an invalid residency snapshot");
        }
        const auto configured_participant_projection = MoEExpertOwnerMap::declaredParticipants(
            *config_.placement_plan,{.reject_tensor_sharded_domains=false});
        const auto &published_participants =
            snapshot->originalOwnerMap().participants();
        const auto &configured_participants =
            configured_participant_projection;
        const bool same_topology =
            snapshot->placement_plan->first_model_layer ==
                config_.placement_plan->first_model_layer &&
            snapshot->placement_plan->placementLayerCapacity() ==
                config_.placement_plan->placementLayerCapacity() &&
            published_participants.size() == configured_participants.size() &&
            std::equal(
                published_participants.begin(),
                published_participants.end(),
                configured_participants.begin(),
                [](const auto &published, const auto &configured)
                {
                    return published.participant_id ==
                               configured.participant_id &&
                           published.tier_idx == configured.tier_idx &&
                           published.tier_name == configured.tier_name &&
                           published.domain_name == configured.domain_name &&
                           published.domain_participant_index ==
                               configured.domain_participant_index &&
                           published.address == configured.address &&
                           published.world_rank == configured.world_rank &&
                           published.world_rank_known ==
                               configured.world_rank_known;
                });
        if (!same_topology)
        {
            throw std::invalid_argument(
                "MoE overlay participant runner authority topology differs from its graph plan");
        }
        owner_map_ = snapshot->jointOwnerMap();
        if(!owner_map_)owner_map_=std::make_shared<const MoEExpertOwnerMap>(snapshot->originalOwnerMap());
        if (config_.ownership == MoEOverlayParticipantGraphOwnership::RankLocalAuxiliary)
        {
            const auto admitted = MoEOverlayHostServiceContract::resolveAdmitted(
                *config_.placement_plan, *owner_map_, *config_.mpi_context);
            for (const int id : admitted.externalParticipants())
            {
                const auto *participant = owner_map_->participantForId(id);
                if (participant && participant->world_rank_known &&
                    participant->world_rank == config_.mpi_context->rank() &&
                    participant->device.is_gpu())
                    admitted_rank_local_endpoint_ids_.push_back(id);
            }
            std::sort(admitted_rank_local_endpoint_ids_.begin(),
                      admitted_rank_local_endpoint_ids_.end());
        }
        for (const auto &participant : owner_map_->participants())
        {
            if (participant.world_rank_known &&
                participant.world_rank == config_.mpi_context->rank() &&
                (config_.ownership == MoEOverlayParticipantGraphOwnership::ExpertOnlyRank ||
                 (participant.domain_name != config_.placement_plan->continuation_domain &&
                  participant.device.is_gpu())))
            {
                local_participants_.push_back(&participant);
            }
        }

        std::sort(
            local_participants_.begin(),
            local_participants_.end(),
            [](const auto *left, const auto *right)
            {
                return left->participant_id < right->participant_id;
            });
        const auto duplicate = std::adjacent_find(
            local_participants_.begin(),
            local_participants_.end(),
            [](const auto *left, const auto *right)
            {
                return left->participant_id == right->participant_id;
            });
        if (duplicate != local_participants_.end())
        {
            throw std::invalid_argument(
                "MoE overlay participant rank resolves duplicate global "
                "participant id " +
                std::to_string((*duplicate)->participant_id));
        }

        if (config_.ownership == MoEOverlayParticipantGraphOwnership::RankLocalAuxiliary)
        {
            std::vector<DeviceId> devices;
            for (const auto *participant : local_participants_)
            {
                if (std::find(devices.begin(), devices.end(), participant->device) != devices.end())
                    throw std::invalid_argument(
                        "Rank-local expert endpoints require one original participant parent per physical GPU");
                devices.push_back(participant->device);
            }
        }
        requireOriginalRankRegistry(*owner_map_, *config_.participant_residency,
                                    config_.mpi_context->rank(), snapshot->epoch);

        /*
         * A RelayOnly rank intentionally owns no endpoint but must enter every
         * matched MPI boundary once. All other auxiliary roles must resolve at
         * least one concrete participant from the canonical owner map.
         */
        if (local_participants_.empty() &&
            execution_plan_->currentRankPlan().role !=
                OverlayRankRole::RelayOnly)
        {
            throw std::invalid_argument(
                "MoE overlay auxiliary rank " +
                std::to_string(config_.mpi_context->rank()) +
                " owns no routed participant endpoints");
        }
    }

    /**
     * @brief Partition raw GGUF blocks into main and learned NextN graph families.
     *
     * The raw Qwen block count includes the trailing NextN block, while the
     * main model graph must stop before it. The trailing block keeps its raw
     * layer number everywhere that matters—prepared-weight lookup, residency
     * bank indexing, and sparse wire keys. This method therefore records it as
     * a separate graph source instead of aliasing it onto an ordinary layer.
     */
    void MoEOverlayParticipantGraphRunner::resolveGraphFamilies()
    {
        const auto &loader = config_.model_context->concreteLoader();
        const int raw_layer_count =
            config_.model_context->totalBlockCount();
        transaction_graph_family_ =
            resolveMoEOverlayInferenceGraphFamilyIdentity(
                loader,
                architecture_,
                raw_layer_count,
                config_.mtp_graph_family_policy,
                /*graph_family_generation=*/1,
                config_.max_graph_activation_rows,
                config_.max_decode_activation_rows,
                config_.max_request_count,
                config_.max_mtp_draft_depth).forRoutedLayerInterval(
                    config_.placement_plan->first_model_layer,
                    config_.placement_plan->placementLayerCapacity());
        main_layer_count_ = transaction_graph_family_.main_layer_count;
        mtp_source_layers_ = transaction_graph_family_.mtp_source_layers;
        routed_layer_capacity_ =
            transaction_graph_family_.routedLayerCapacity();

        /* The model-frozen placement plan is the topology-wide storage
         * authority. The independently resolved graph family describes the
         * exact graphs this follower will construct. Requiring equality here
         * prevents a raw GGUF sidecar block from silently enlarging only one
         * rank's runtime table, and catches a missing MTP bank before capture. */
        const int planned_layer_capacity =
            config_.placement_plan->placementLayerCapacity();
        if (routed_layer_capacity_ <= 0 ||
            planned_layer_capacity != routed_layer_capacity_)
        {
            throw std::invalid_argument(
                "MoE overlay participant graph-family layer capacity " +
                std::to_string(routed_layer_capacity_) +
                " disagrees with frozen placement capacity " +
                std::to_string(planned_layer_capacity));
        }

        const int num_experts = loader.getInt(
            architecture_ + ".expert_count", 0);
        if (num_experts <= 0)
        {
            throw std::runtime_error(
                "MoE participant graph has no routed-expert geometry");
        }
        // Source recipes precede preparation. A jointly owned expert has one
        // logical placement and every original member's source rectangle;
        // requiring a scalar whole owner would erase that distinction.
        requireMoEOverlayMTPSourceCoverage(*owner_map_,
            config_.participant_residency?config_.participant_residency->projectionPreparation().get():nullptr,
            mtp_source_layers_,num_experts,config_.mpi_context->rank());

        if (config_.ownership == MoEOverlayParticipantGraphOwnership::RankLocalAuxiliary)
        {
            std::vector<int> targets;
            for (const auto &participant : owner_map_->participants())
            {
                if (participant.world_rank_known &&
                    participant.world_rank == config_.mpi_context->rank() &&
                    participant.device.is_gpu() &&
                    participant.domain_name != config_.placement_plan->continuation_domain)
                    targets.push_back(participant.participant_id);
            }
            std::sort(targets.begin(), targets.end());
            transaction_topology_identity_ = makeMoEOverlayRankLocalInferenceTopologyIdentity(
                *owner_map_, transaction_graph_family_,
                config_.placement_plan->continuation_domain_spec.logical_root_participant,
                targets);
        }
        else
        transaction_topology_identity_ =
            makeMoEOverlayInferenceTopologyIdentity(
                *owner_map_,
                transaction_graph_family_,
                execution_plan_->continuation_root_rank,
                config_.mpi_context->rank());
    }

    /**
     * @brief Establish one planner-bounded compact tensor family per local endpoint.
     *
     * A routed local-expert stage packs each input row once and retains its
     * participant-local routes on the tensor's top-k axis. Allocate a complete
     * row-family manifest published by the distributed schedule before graph
     * construction. Small power-of-two decode/MTP shapes are added to that
     * exact prefill ladder. Every layer keeps stable addresses, while a live
     * decode/verifier packet transfers one hidden and one aggregate output row
     * rather than repeating both payloads for every selected expert.
     *
     * Relay-only ranks intentionally produce no arena. Every declared local
     * endpoint produces exactly one arena even when its epoch-one resident
     * mask is empty. Residency migration can make that endpoint live later,
     * and captured topology may not be allocated or rebound at that point.
     * Graph construction treats a missing entry as a fatal ownership
     * violation rather than allocating a stage-private substitute.
     */
    void MoEOverlayParticipantGraphRunner::createSerialCompactBufferArenas()
    {
        const int d_model = config_.model_context->embeddingLength();
        const auto &loader = config_.model_context->concreteLoader();
        const std::string &architecture = config_.model_context->architecture();
        const int num_experts = loader.getInt(architecture + ".expert_count", 0);
        const int routing_top_k =
            loader.getInt(architecture + ".expert_used_count", 0);
        int expert_intermediate =
            loader.getInt(
                architecture + ".expert_feed_forward_length", 0);
        if (expert_intermediate <= 0)
            expert_intermediate = config_.model_context->feedForwardLength();
        if (d_model <= 0 || num_experts <= 0 || routing_top_k <= 0 ||
            expert_intermediate <= 0)
        {
            throw std::runtime_error(
                "Participant serial compact arena could not resolve model routing geometry");
        }

        const size_t row_capacity =
            static_cast<size_t>(config_.max_graph_activation_rows);
        if (row_capacity == 0)
        {
            throw std::runtime_error(
                "Participant serial compact arena resolved zero row capacity");
        }

        for (const auto *participant : local_participants_)
        {
            if (!participant)
                throw std::logic_error(
                    "Participant serial compact arena received a null endpoint");

            MoELocalExpertSerialBufferArena::Config arena_config;
            arena_config.device_id = participant->device;
            arena_config.row_capacity = row_capacity;
            arena_config.row_capacity_buckets =
                MoELocalExpertSerialBufferArena::powerOfTwoRowBucketsThrough(
                    static_cast<size_t>(
                        config_.max_decode_activation_rows));
            /*
             * The continuation ticket carries one of these exact physical
             * shapes. Copy the schedule-owned manifest verbatim so follower
             * capture and admission cannot drift when the production ladder
             * contains non-power-of-two buckets such as 600 rows.
             */
            arena_config.row_capacity_buckets.reserve(
                arena_config.row_capacity_buckets.size() +
                config_.prefill_graph_row_shapes.size());
            for (const int shape : config_.prefill_graph_row_shapes)
            {
                arena_config.row_capacity_buckets.push_back(
                    static_cast<size_t>(shape));
            }
            arena_config.d_model = d_model;
            arena_config.routing_top_k = routing_top_k;
            if (participant->device.is_cpu())
            {
                const auto manager=config_.model_context->concreteWeightManager();
                const auto memory=manager?manager->physicalMemoryAuthority():nullptr;
                if(!memory) throw std::logic_error("CPU follower serial roots require the existing model PMA");
                const auto &admission=requireCPUSerialExpertStorageAdmission(config_.cpu_serial_expert_storage,
                    {memory->worldRank(),participant->device},participant->participant_id,participant->device);
                arena_config.cpu_storage_layout=admission.envelope->project(config_.prefill_graph_row_capacity,
                    config_.max_decode_activation_rows,config_.prefill_graph_row_shapes);
                arena_config.row_capacity=arena_config.cpu_storage_layout->maximumRows();
                arena_config.row_capacity_buckets.clear();
                for(const auto &family:arena_config.cpu_storage_layout->families())
                    arena_config.row_capacity_buckets.push_back(family.rows);
                arena_config.physical_memory_authority=memory;
                arena_config.cpu_canonical_route_storage =
                    MoELocalExpertSerialBufferArena::
                        CPUCanonicalRouteStoragePolicy::RetainSerialMaximum;
                if(arena_config.cpu_storage_layout->jointRoutes())
                    arena_config.cpu_canonical_route_storage=arena_config.cpu_storage_layout->jointRoutes()->isRoot()
                        ?MoELocalExpertSerialBufferArena::CPUCanonicalRouteStoragePolicy::RetainSerialMaximum:
                         MoELocalExpertSerialBufferArena::CPUCanonicalRouteStoragePolicy::Disabled;
                arena_config.cpu_grouped_scratch_storage =
                    MoELocalExpertSerialBufferArena::
                        CPUGroupedScratchStoragePolicy::RetainSerialMaximum;
                arena_config.num_experts = num_experts;
                arena_config.expert_intermediate = arena_config.cpu_storage_layout->jointRoutes()
                    ?arena_config.cpu_storage_layout->jointRoutes()->intermediate():expert_intermediate;
            }
            arena_config.logical_participant_id = participant->participant_id;
            arena_config.debug_name =
                "moe_overlay_participant_serial_compact.p" +
                std::to_string(participant->participant_id) + "@" +
                participant->device.to_string();

            auto arena = std::make_shared<MoELocalExpertSerialBufferArena>(
                std::move(arena_config));
            const bool inserted = serial_compact_buffer_arenas_.emplace(
                participant->participant_id, std::move(arena)).second;
            if (!inserted)
            {
                throw std::logic_error(
                    "Participant serial compact arena has duplicate logical endpoint p" +
                    std::to_string(participant->participant_id));
            }

        }
    }

    /**
     * @brief Return the only compact tensor owner legal for @p participant.
     *
     * This extra validation is intentionally at graph materialization rather
     * than hidden in a map lookup.  It catches a stale owner map, a malformed
     * participant id, or an accidental device reassignment before either graph
     * can capture the wrong tensor address.
     */
    std::shared_ptr<MoELocalExpertSerialBufferArena>
    MoEOverlayParticipantGraphRunner::serialCompactBufferArenaForParticipant(
        const MoEExpertOwnerParticipant &participant) const
    {
        const auto found = serial_compact_buffer_arenas_.find(
            participant.participant_id);
        if (found == serial_compact_buffer_arenas_.end() || !found->second)
        {
            throw std::logic_error(
                "Participant graph is missing its immutable serial compact arena for p" +
                std::to_string(participant.participant_id));
        }
        const auto &arena = found->second;
        if (arena->deviceId() != participant.device ||
            !arena->logicalParticipantId() ||
            *arena->logicalParticipantId() != participant.participant_id)
        {
            throw std::logic_error(
                "Participant graph compact arena ownership does not match p" +
                std::to_string(participant.participant_id));
        }
        return arena;
    }

    /**
     * @brief Materialize and register exact real-weight GEMM engines for local experts.
     *
     * Weight preparation happens before graph construction and uses the
     * rank-specific map above.  A later local stage is registry-only; it cannot
     * fall back to a raw model parent or prepare a new kernel on the hot path.
     */
    void MoEOverlayParticipantGraphRunner::prepareParticipantWeights()
    {
        auto weight_manager = config_.model_context->concreteWeightManager();
        if (!weight_manager)
            throw std::runtime_error("Participant model context has no WeightManager");

        InferenceStrategy strategy;
        strategy.mode = WeightInferenceMode::ExpertOverlayRank;
        strategy.model_id = ModelContextId{
            reinterpret_cast<uint64_t>(config_.model_context.get())};
        strategy.devices = participantDevices(local_participants_);
        WeightPlan plan(std::move(strategy));
        if (config_.prepared_weight_admission ==
                PreparedWeightAdmission::AllocateCompleteSet &&
            !local_participants_.empty())
        {
            ScopedWeightLoadDetailTimer timer(
                "overlay.weights.build_plan");
            const auto original_preparation=config_.participant_residency->projectionPreparation();
            if(!original_preparation && residency_authority_->snapshot()->jointOwnerMap())
                throw std::runtime_error("CPU joint participant weights lost original admitted source preparation");
            const auto preparation = (original_preparation?*original_preparation:
                MoEExpertOverlayPreparationPlan::build(*runtime_plan_,config_.model_context->concreteLoader()))
                .filteredForRank(execution_plan_->currentRankPlan())
                .filteredForDevices(participantDevices(local_participants_));
            if(preparation.jointOwnerMap() && preparation.jointOwnerMap().get()!=owner_map_.get())
                throw std::runtime_error("CPU joint participant preparation replaced original source owner map");
            const auto sources = preparation.sourceWeightPlan(
                config_.model_context->concreteLoader(), plan.strategy().model_id);
            for (const auto &requirement : sources.requirements())
                plan.add(requirement);
        }
        const ModelContextId model_id = plan.strategy().model_id;

        {
            ScopedWeightLoadDetailTimer timer(
                "overlay.weights.bind_prepared_store");
            prepared_store_ = weight_manager->preparedWeightStoreIfInitialized();
            if (!prepared_store_)
            {
                if (config_.prepared_weight_admission ==
                    PreparedWeightAdmission::ReuseCertifiedCompleteSet)
                {
                    throw std::runtime_error(
                        "Certified participant-weight reuse lost the exact "
                        "model-owned PreparedWeightStore");
                }
                prepared_store_ =
                    std::make_shared<PreparedWeightStore>(model_id);
                weight_manager->setPreparedWeightStore(prepared_store_);
            }
            if (!prepared_store_->bindModelIdIfUnset(model_id))
            {
                throw std::runtime_error(
                    "Participant PreparedWeightStore belongs to another model");
            }
        }

        if (plan.empty() &&
            config_.prepared_weight_admission ==
                PreparedWeightAdmission::AllocateCompleteSet)
        {
            /* Relay ranks own protocol state but no expert weight authority. */
            return;
        }

        if (!plan.empty())
        {
            ScopedWeightLoadDetailTimer timer(
                "overlay.weights.materialize_plan");
            frozen_weights_ = std::make_unique<FrozenModelWeightSet>(
                weight_manager->materialize(plan));
        }
        for (const DeviceId device : participantDevices(local_participants_))
        {
            if (!weight_manager->prepareMoEExpertOverlayWeights(
                    *runtime_plan_,
                    device,
                    frozen_weights_.get(),
                    execution_plan_.get(),
                    config_.prepared_weight_admission,
                    config_.participant_residency?config_.participant_residency->projectionPreparation():nullptr))
            {
                throw std::runtime_error(
                    "Failed to prepare exact MoE overlay participant weights on " +
                    device.to_string());
            }
        }
    }

    /**
     * @brief Create the exact contexts that execute locally owned graph nodes.
     *
     * CPU transport stages always need a host context.  GPU contexts are
     * created once here, before graph materialization, so local-expert stages
     * can bind their exact worker stream and persistent arena storage without
     * consulting a backend default stream during inference.
     */
    void MoEOverlayParticipantGraphRunner::createDeviceContexts()
    {
        std::vector<DeviceId> devices = participantDevices(local_participants_);
        if (std::find(devices.begin(), devices.end(), DeviceId::cpu()) ==
            devices.end())
        {
            /* Every graph has CPU transport boundary stages. */
            devices.push_back(DeviceId::cpu());
        }

        for (const DeviceId device : devices)
        {
            auto context = IDeviceContext::create(device);
            if (!context)
            {
                throw std::runtime_error(
                    "Failed to create participant context for " +
                    device.to_string());
            }
            auto *raw_context = context.get();
            auto [it, inserted] = owned_device_contexts_.emplace(
                device, std::move(context));
            if (!inserted)
            {
                throw std::runtime_error(
                    "Duplicate participant device context for " +
                    device.to_string());
            }
            execution_contexts_.emplace(device, raw_context);
        }
    }

    /**
     * @brief Materialize one shared placement/runtime family per follower GPU.
     *
     * Policy ownership and GPU execution-state ownership are different axes.
     * Every GPU receives a real mirrored runtime table and epoch ticket. Under
     * host-resident heterogeneous authority, background publication writes its
     * inactive bank through the table's retained host recipe. Under all-GPU
     * device authority, the same arena additionally consumes the mapped global
     * admission word. Row-shape and MTP graphs borrow these addresses; none may
     * create a private placement generation.
     */
    void MoEOverlayParticipantGraphRunner::createParticipantGpuRuntimes()
    {
        const auto &loader = config_.model_context->concreteLoader();
        const std::string &arch = config_.model_context->architecture();
        /* resolveGraphFamilies() already proved this value against the frozen
         * placement plan. Never re-derive it from totalBlockCount(): Qwen GGUFs
         * include an inactive trailing NextN block when MTP is disabled. */
        const int runtime_layer_count = routed_layer_capacity_;
        const int num_experts = loader.getInt(arch + ".expert_count", 0);
        const int top_k = loader.getInt(arch + ".expert_used_count", 0);
        const auto authority_execution =
            config_.placement_plan->authority_execution;
        if (runtime_layer_count <= 0 || num_experts <= 0 || top_k <= 0 ||
            authority_execution ==
                MoEOverlayAuthorityExecutionKind::Unresolved)
        {
            throw std::invalid_argument(
                "Mapped follower GPU runtime has unresolved model geometry or policy authority");
        }

        const bool device_resident_authority =
            authority_execution ==
            MoEOverlayAuthorityExecutionKind::DeviceResident;
        if (device_resident_authority !=
            static_cast<bool>(config_.device_controller_fabric))
        {
            throw std::invalid_argument(
                device_resident_authority
                    ? "Device-resident follower GPU runtime has no topology-wide controller fabric"
                    : "Host-resident follower GPU runtime unexpectedly received a device-controller fabric");
        }

        std::vector<int> expected_gpu_participants;
        for (const auto *participant : local_participants_)
        {
            if (participant && participant->device.is_gpu())
                expected_gpu_participants.push_back(participant->participant_id);
        }
        if (config_.device_controller_fabric)
        {
            const auto &layout = config_.device_controller_fabric->layout();
            const auto &header = layout.header;
            if (!layout.valid() ||
                header.first_model_layer != transaction_graph_family_.first_model_layer ||
                header.num_layers !=
                    static_cast<std::uint32_t>(runtime_layer_count) ||
                header.num_experts !=
                    static_cast<std::uint32_t>(num_experts))
            {
                throw std::invalid_argument(
                    "Mapped follower GPU runtime geometry disagrees with the topology-wide controller fabric");
            }
            const auto snapshot = residency_authority_->snapshot();
            if (!snapshot || !snapshot->valid())
                throw std::invalid_argument(
                    "Mapped follower GPU ownership has no original residency snapshot");
            requireMoEOverlayParticipantGPUOwnership(
                *snapshot->placement_plan, snapshot->originalOwnerMap(),
                config_.mpi_context->rank(), config_.ownership,
                config_.device_controller_fabric->localParticipantIds(),
                expected_gpu_participants);
        }

        for (const auto *participant : local_participants_)
        {
            if (!participant || !participant->device.is_gpu())
                continue;

            std::optional<MoEOverlayDeviceControllerParticipantBinding>
                controller_binding;
            if (config_.device_controller_fabric)
            {
                controller_binding =
                    config_.device_controller_fabric->participantBinding(
                        participant->participant_id);
                if (!controller_binding->valid() ||
                    controller_binding->participant_id !=
                        participant->participant_id ||
                    controller_binding->device != participant->device ||
                    !controller_binding->controller ||
                    !controller_binding->lifetime)
                {
                    throw std::logic_error(
                        "Mapped follower received an incomplete controller-fabric participant binding");
                }
            }

            auto domain_participants =
                owner_map_->participantIdsForTier(participant->tier_idx);
            if (domain_participants.empty() ||
                domain_participants.size() > kDeviceMoEMaxParticipants ||
                participant->domain_participant_index < 0 ||
                static_cast<std::size_t>(
                    participant->domain_participant_index) >=
                    domain_participants.size())
            {
                throw std::invalid_argument(
                    "Mapped follower domain is outside the device runtime participant ABI");
            }
            std::vector<std::uint8_t> dense_domain_ids(
                domain_participants.size(), 0u);
            for (const int global_participant_id : domain_participants)
            {
                const auto *const member =
                    owner_map_->participantForId(global_participant_id);
                if (!member || member->tier_idx != participant->tier_idx ||
                    member->domain_name != participant->domain_name ||
                    member->domain_participant_index < 0 ||
                    static_cast<std::size_t>(
                        member->domain_participant_index) >=
                        dense_domain_ids.size() ||
                    dense_domain_ids[static_cast<std::size_t>(
                        member->domain_participant_index)] != 0u)
                {
                    throw std::invalid_argument(
                        "Mapped follower tier does not provide a dense unique domain participant namespace");
                }
                dense_domain_ids[static_cast<std::size_t>(
                    member->domain_participant_index)] = 1u;
            }

            const bool collect_dynamic_service_telemetry =
                config_.durable_maintenance_policy ==
                MoEOverlayDurableMaintenancePolicy::DynamicPlacement;
            const auto service_telemetry_catalog =
                collect_dynamic_service_telemetry &&
                        config_.residency_authority
                    ? config_.residency_authority->economyLayerCatalog()
                    : nullptr;
            if (collect_dynamic_service_telemetry &&
                !service_telemetry_catalog)
            {
                throw std::logic_error(
                    "Dynamic mapped follower has no canonical economy layer catalog");
            }

            auto runtime = std::make_unique<ParticipantGpuRuntime>();
            runtime->participant_id = participant->participant_id;
            runtime->device = participant->device;
            runtime->domain_participant_id = static_cast<std::uint32_t>(
                participant->domain_participant_index);
            runtime->domain_participant_count = static_cast<std::uint32_t>(
                domain_participants.size());
            runtime->authority_execution = authority_execution;
            runtime->controller_binding = controller_binding;
            runtime->worker = participantWorkerContext(participant->device);
            if (!runtime->worker)
            {
                throw std::runtime_error(
                    "Mapped follower runtime could not resolve its exact GPU worker");
            }

            runtime->worker->submitAndWait(
                [&]
                {
                    runtime->runtime_publication_stream =
                        runtime->worker->defaultStream();
                    IBackend *const backend = getBackendFor(
                        participant->device);
                    if (!backend || !runtime->runtime_publication_stream)
                    {
                        throw std::runtime_error(
                            "Mapped follower runtime could not resolve its "
                            "initial-publication backend or exact stream");
                    }
                    const int ordinal = participant->device.gpu_ordinal();
                    const auto make_event =
                        [backend, ordinal]() -> std::shared_ptr<void>
                    {
                        void *const event = backend->createEvent(ordinal);
                        if (!event)
                            return {};
                        return std::shared_ptr<void>(
                            event,
                            [backend, ordinal](void *owned)
                            {
                                if (owned)
                                    backend->destroyEvent(owned, ordinal);
                            });
                    };
                    runtime->initial_runtime_source_ready_event =
                        make_event();
                    runtime->initial_runtime_published_event = make_event();
                    runtime->route_scratch = std::make_shared<
                        DeviceMoESerialRouteScratchArena>(
                        DeviceMoESerialRouteScratchArena::Config{
                            .device_id = participant->device,
                            .num_experts = num_experts,
                            .top_k = top_k,
                            .token_capacity =
                                config_.max_graph_activation_rows,
                        });
                    DeviceMoEOverlayEpochArena::Config epoch_config{
                        .device_id = participant->device,
                        .initial_epoch = 1u,
                        .initial_bank = 1u,
                        .request_slot_capacity = 1u,
                    };
                    if (controller_binding)
                    {
                        epoch_config.external_admission_epoch =
                            &controller_binding->controller->admission_epoch;
                        epoch_config.external_admission_lifetime =
                            controller_binding->lifetime;
                    }
                    epoch_config.memory_authority = config_.model_context->concreteWeightManager()->physicalMemoryAuthority();
                    runtime->epoch_arena = std::make_shared<
                        DeviceMoEOverlayEpochArena>(
                        std::move(epoch_config));
                    runtime->runtime_table = std::make_unique<
                        DeviceMoERuntimeTable>(
                        DeviceMoERuntimeTable::Config{
                            .device_id = participant->device,
                            .num_layers = runtime_layer_count,
                            .num_experts = num_experts,
                            .top_k = top_k,
                            .mirror_to_device = true,
                            .overlay_service_telemetry_coverage =
                                collect_dynamic_service_telemetry
                                    ? MoEOverlayServiceTelemetryCoverage::
                                          CatalogStratifiedSample
                                    : MoEOverlayServiceTelemetryCoverage::
                                          Disabled,
                            .overlay_service_telemetry_catalog =
                                service_telemetry_catalog,
                            .prefill_token_capacity =
                                config_.max_graph_activation_rows,
                            .deferred_verifier_token_capacity =
                                config_.max_decode_activation_rows,
                            .serial_route_scratch_arena =
                                runtime->route_scratch,
                            .overlay_epoch_arena = runtime->epoch_arena,
                            .overlay_epoch_ticket_slot = 0u,
                            .first_model_layer = transaction_graph_family_.first_model_layer,
                        });
                });
            if (!runtime->route_scratch || !runtime->epoch_arena ||
                !runtime->runtime_table ||
                !runtime->runtime_publication_stream ||
                !runtime->initial_runtime_source_ready_event ||
                !runtime->initial_runtime_published_event)
            {
                throw std::runtime_error(
                    "Mapped follower runtime allocation returned incomplete model-lifetime state");
            }
            runtime->initialized_layers.assign(
                static_cast<std::size_t>(runtime_layer_count), 0u);

            const auto [_, inserted] =
                participant_gpu_runtimes_.emplace(
                    participant->participant_id, std::move(runtime));
            if (!inserted)
            {
                throw std::logic_error(
                    "Mapped follower constructed duplicate controller runtime authority");
            }

            PerfStatsCollector::addCounter(
                "moe_overlay_controller",
                "follower_runtime_tables_materialized",
                1.0,
                "model_setup",
                participant->device.toString(),
                {{"participant",
                  std::to_string(participant->participant_id)},
                 {"domain_participant",
                  std::to_string(participant->domain_participant_index)},
                 {"domain_participants",
                  std::to_string(domain_participants.size())},
                 {"layers", std::to_string(runtime_layer_count)},
                 {"blocking_hot_path", "false"},
                 {"authority_execution",
                  std::string(toString(authority_execution))},
                 {"mapped_device_controller",
                  controller_binding ? "true" : "false"}});
        }
    }

    /** @brief Resolve the exact model-lifetime runtime for one global participant. */
    MoEOverlayParticipantGraphRunner::ParticipantGpuRuntime &
    MoEOverlayParticipantGraphRunner::participantGpuRuntimeForParticipant(
        int participant_id) const
    {
        const auto found = participant_gpu_runtimes_.find(participant_id);
        if (found == participant_gpu_runtimes_.end() || !found->second)
        {
            throw std::out_of_range(
                "Mapped follower has no GPU placement runtime for participant " +
                std::to_string(participant_id));
        }
        return *found->second;
    }

    /**
     * @brief Release mapped raw GGUF bytes once prepared engines own the selected experts.
     *
     * The frozen bindings retain metadata and packed-engine lifetimes while the
     * raw host pages are no longer an execution authority.  A failure to
     * release one mapping is diagnostic-only because it cannot change the
     * prepared graph's numerical result or ownership.
     */
    void MoEOverlayParticipantGraphRunner::releasePreparedSourceBytes()
    {
        if (!frozen_weights_)
            return;

        const auto manager = config_.model_context->concreteWeightManager();
        const auto *metadata = manager->weightMetadataRegistry();
        if (!metadata)
            throw std::logic_error("Participant source retirement requires its original weight metadata");
        const auto released_bytes = retireMaterializedParticipantSources(*frozen_weights_, *metadata);

        PerfStatsCollector::addCounter(
            "moe_overlay_participant_graph",
            "released_source_bytes",
            static_cast<double>(released_bytes),
            "setup",
            participantDeviceList(local_participants_),
            {{"participants", participantIdList(local_participants_)}});
    }

    /**
     * @brief Validate one live packet shape and return the setup-owned graph.
     *
     * Every possible row count at or below the planner admission shares this
     * immutable graph. The per-transaction sparse views contain the live row
     * count, so no tail-shaped graph construction is legal in the hot path.
     */
    MoEOverlayParticipantGraphRunner::CachedParticipantGraph &
    MoEOverlayParticipantGraphRunner::graphForRows(int logical_rows)
    {
        if (logical_rows <= 0 ||
            logical_rows > config_.max_graph_activation_rows)
        {
            throw std::invalid_argument(
                "Participant live rows " + std::to_string(logical_rows) +
                " exceed immutable planner-admitted graph capacity " +
                std::to_string(config_.max_graph_activation_rows));
        }
        if (!main_graph_ || !main_graph_->graph)
        {
            throw std::logic_error(
                "Participant main graph was not materialized during setup");
        }
        return *main_graph_;
    }

    /**
     * @brief Build one ordered sparse dispatch/local-expert/return graph variant.
     *
     * This graph contains every globally ordered remote MPI boundary so it
     * remains protocol-symmetric with the dense continuation root graph.
     * Participants in a multi-device continuation domain execute in their own
     * captured LocalTP graphs and therefore never enter the rank-level MPI
     * protocol. Only remote target participants owned by this rank receive a
     * local compute stage. Those stages bind their pre-existing arena; graph
     * construction is forbidden from allocating a layer-private compact
     * packet.
     */
    std::unique_ptr<MoEOverlayParticipantGraphRunner::CachedParticipantGraph>
    MoEOverlayParticipantGraphRunner::buildGraph(
        const ParticipantGraphBuildSpec &spec)
    {
        const bool valid_main =
            spec.role == ParticipantGraphFamilyRole::Main &&
            spec.mtp_graph_depth == -1 &&
            spec.row_capacity == config_.max_graph_activation_rows;
        const bool valid_mtp =
            spec.role == ParticipantGraphFamilyRole::MTPDraft &&
            spec.mtp_graph_depth >= 0 &&
            spec.row_capacity == config_.max_decode_activation_rows;
        if ((!valid_main && !valid_mtp) || spec.source_layers.empty())
        {
            throw std::invalid_argument(
                "Participant retained graph spec has an invalid role, depth, layer family, or admitted capacity");
        }
        const int row_capacity = spec.row_capacity;

        auto result = std::make_unique<CachedParticipantGraph>();
        result->graph = std::make_unique<ComputeGraph>();
        result->role = spec.role;
        result->mtp_graph_depth = spec.mtp_graph_depth;
        result->row_capacity = row_capacity;
        result->source_layers = spec.source_layers;
        const int d_model = config_.model_context->embeddingLength();
        const auto &loader = config_.model_context->concreteLoader();
        const std::string &arch = config_.model_context->architecture();
        const int num_experts = loader.getInt(arch + ".expert_count", 0);
        const int top_k = loader.getInt(arch + ".expert_used_count", 0);
        int expert_intermediate =
            loader.getInt(arch + ".expert_feed_forward_length", 0);
        if (expert_intermediate <= 0)
            expert_intermediate = config_.model_context->feedForwardLength();
        if (d_model <= 0 || num_experts <= 0 || top_k <= 0 ||
            expert_intermediate <= 0)
        {
            throw std::runtime_error(
                "Participant graph could not resolve Qwen MoE geometry");
        }

        const std::vector<int> local_participant_ids = participantIds();
        const auto weight_manager =
            config_.model_context->concreteWeightManager();
        if (!weight_manager)
        {
            throw std::logic_error(
                "Participant graph has no model-owned WeightManager");
        }
        if (!weight_manager->physicalMemoryAuthority() ||
            !serial_graph_family_workspace_allocator_)
        {
            throw std::logic_error(
                "Participant graph has no runner-owned serial workspace authority");
        }
        auto &registry = weight_manager->expertGemmRegistry();
        const int continuation_root =
            config_.placement_plan->continuation_domain_spec
                .logical_root_participant;
        const auto *const declared_continuation_root =
            owner_map_->participantForId(continuation_root);
        if (continuation_root < 0 || !declared_continuation_root ||
            declared_continuation_root->domain_name !=
                config_.placement_plan->continuation_domain ||
            !declared_continuation_root->world_rank_known ||
            declared_continuation_root->world_rank !=
                execution_plan_->continuation_root_rank)
        {
            throw std::runtime_error(
                "Participant graph could not resolve the planner-declared continuation root participant");
        }
        const int local_world_rank = config_.mpi_context->rank();
        const int source_world_rank =
            execution_plan_->continuation_root_rank;
        if (local_world_rank == source_world_rank &&
            config_.ownership != MoEOverlayParticipantGraphOwnership::RankLocalAuxiliary)
        {
            throw std::logic_error(
                "Remote ExpertOverlay participant graph cannot run on the continuation authority rank");
        }

        const size_t row_capacity_size =
            static_cast<size_t>(row_capacity);
        const size_t top_k_size = static_cast<size_t>(top_k);
        if (row_capacity_size >
            std::numeric_limits<size_t>::max() / top_k_size)
        {
            throw std::overflow_error(
                "Participant graph sparse entry capacity overflows size_t");
        }
        const size_t entry_capacity =
            row_capacity_size * top_k_size;

        /*
         * Rank transport identity belongs to the complete serial graph family,
         * not to the currently selected local compute specialization. The
         * continuation graph owns one maximum-capacity channel reused by main,
         * decode, and every MTP depth. A draft endpoint may execute a four-row
         * retained graph inside that channel, but deriving its POSIX mapping
         * from four rows would make it wait on a different object from the
         * continuation's prefill-sized mapping. Keep local graph admission at
         * `row_capacity_size` while both ranks derive transport storage from the
         * planner-published family maximum.
         */
        const size_t transport_row_capacity_size =
            static_cast<size_t>(config_.max_graph_activation_rows);
        if (transport_row_capacity_size == 0 ||
            transport_row_capacity_size < row_capacity_size ||
            transport_row_capacity_size >
                std::numeric_limits<size_t>::max() / top_k_size)
        {
            throw std::overflow_error(
                "Participant graph rank-batch family capacity is invalid");
        }
        const size_t transport_entry_capacity =
            transport_row_capacity_size * top_k_size;

        /*
         * A batch receives all local endpoint packets at once. Each endpoint
         * therefore owns one independent packet family, while layers reuse that
         * endpoint family serially. This is the smallest setup-only allocation
         * compatible with concurrent participant device streams.
         */
        std::unordered_map<
            int,
            std::shared_ptr<MoEOverlayCollectiveWorkspace>>
            protocol_workspaces;
        const auto workspace_for_participant =
            [&](int participant)
                -> std::shared_ptr<MoEOverlayCollectiveWorkspace>
        {
            const auto existing =
                protocol_workspaces.find(participant);
            if (existing != protocol_workspaces.end())
                return existing->second;
            auto workspace =
                std::make_shared<MoEOverlayCollectiveWorkspace>(
                    MoEOverlayCollectiveWorkspace::FixedCapacityConfig{
                        .max_rows = row_capacity_size,
                        .max_entries = entry_capacity,
                        .d_model = d_model,
                        .top_k = top_k,
                        .device = DeviceId::cpu(),
                        .reuse_policy =
                            MoEOverlayCollectiveWorkspace::
                                StorageReusePolicy::SerialGraphFamily,
                    });
            protocol_workspaces.emplace(participant, workspace);
            return workspace;
        };

        const auto domain_ordinal =
            [&](const std::string &domain_name)
        {
            for (size_t index = 0;
                 index < config_.placement_plan->domains.size();
                 ++index)
            {
                if (config_.placement_plan->domains[index].name ==
                    domain_name)
                {
                    return static_cast<int>(index);
                }
            }
            throw std::logic_error(
                "Participant rank batch has no stable routed-domain ordinal for " +
                domain_name);
        };

        const auto *const continuation_descriptor =
            owner_map_->participantForId(continuation_root);
        if (!continuation_descriptor ||
            !continuation_descriptor->world_rank_known ||
            continuation_descriptor->world_rank != source_world_rank ||
            continuation_descriptor->tier_idx < 0 ||
            static_cast<size_t>(continuation_descriptor->tier_idx) >=
                config_.placement_plan->routed_tiers.size())
        {
            throw std::logic_error(
                "Participant activation channel cannot resolve its continuation endpoint topology");
        }
        const MoEOverlayActivationLaneEndpoint continuation_endpoint{
            .world_rank = source_world_rank,
            .participant_id = continuation_root,
            .tier_priority =
                config_.placement_plan
                    ->routed_tiers[static_cast<size_t>(
                        continuation_descriptor->tier_idx)]
                    .priority,
            .domain_ordinal =
                domain_ordinal(continuation_descriptor->domain_name),
        };
        const auto activation_graph_families =
            makeMoEOverlayActivationGraphFamilyManifests(
                transaction_graph_family_);

        /*
         * One transport owns one immutable rank/tier participant group and is
         * shared by every transformer layer. Its fixed replay ledger keys each
         * layer separately; its in-flight guard makes accidental graph overlap
         * a fatal ownership error.
         */
        const auto transport_for_group =
            [&](int tier_index,
                int routed_domain_ordinal,
                const std::vector<int> &participants)
                -> std::shared_ptr<IMoEOverlayRankBatchTransport>
        {
            if (participants.empty())
            {
                throw std::invalid_argument(
                    "Participant rank batch cannot bind an empty group");
            }
            const bool rank_local_endpoints = config_.ownership ==
                MoEOverlayParticipantGraphOwnership::RankLocalAuxiliary;
            const std::string key = rank_local_endpoints
                ? makeMoEOverlayRankLocalGPUChannelIdentity(
                    tier_index, routed_domain_ordinal, source_world_rank, participants)
                : makeMoEOverlayRankBatchChannelIdentity(
                    tier_index, routed_domain_ordinal,
                    source_world_rank, local_world_rank, participants);
            const auto existing =
                rank_batch_transports_.find(key);
            if (existing != rank_batch_transports_.end())
                return existing->second;

            if (rank_local_endpoints ||
                resolveMoEOverlayRankBatchTransportKind(
                    *config_.mpi_context, source_world_rank, local_world_rank) ==
                    MoEOverlayRankBatchTransportKind::NodeLocalSharedRows)
            {
                if (!config_.rank_batch_transport_registry)
                {
                    throw std::logic_error(
                        "Participant graph has no preflight node-local activation-channel registry for " +
                        key);
                }
                auto transport =
                    config_.rank_batch_transport_registry->require(
                        key,
                        source_world_rank,
                        local_world_rank,
                        participants);
                rank_batch_transports_.emplace(key, transport);
                return transport;
            }

            const size_t row_multiplier =
                std::min(top_k_size, participants.size());
            if (transport_row_capacity_size >
                std::numeric_limits<size_t>::max() /
                    row_multiplier)
            {
                throw std::overflow_error(
                    "Participant rank-batch row capacity overflows size_t");
            }
            auto wire_workspace =
                std::make_shared<MoEOverlayRankBatchWireWorkspace>(
                    MoEOverlayRankBatchWireWorkspace::Config{
                        .participant_ids = participants,
                        .max_total_rows =
                            transport_row_capacity_size * row_multiplier,
                        .max_total_entries = transport_entry_capacity,
                        .d_model = d_model,
                        .top_k = top_k,
                    });

            std::vector<MoEOverlayActivationLocalLaneBinding> local_lanes;
            for (const int participant_id : participants)
            {
                const auto *const participant =
                    owner_map_->participantForId(participant_id);
                if (!participant || !participant->world_rank_known ||
                    participant->world_rank != local_world_rank ||
                    participant->tier_idx != tier_index ||
                    domain_ordinal(participant->domain_name) !=
                        routed_domain_ordinal ||
                    !participant->device.is_valid())
                {
                    throw std::logic_error(
                        "Participant activation channel group disagrees with the immutable owner map");
                }
                local_lanes.push_back({
                    .participant_id = participant_id,
                    .device = participant->device,
                });
            }
            const auto *const continuation_participant =
                owner_map_->participantForId(continuation_endpoint.participant_id);
            const bool gpu_to_gpu_prefill_fp16 =
                continuation_participant && continuation_participant->device.is_gpu() &&
                !local_lanes.empty() &&
                std::all_of(
                    local_lanes.begin(),
                    local_lanes.end(),
                    [](const auto &lane) { return lane.device.is_gpu(); });
            auto transport = createMoEOverlayRankBatchTransport(
                    MoEOverlayRankBatchTransportConfig{
                        .mpi_ctx = config_.mpi_context,
                        .physical_memory_authority = config_.model_context->concreteWeightManager()->physicalMemoryAuthority(),
                        .source_world_rank = source_world_rank,
                        .target_world_rank = local_world_rank,
                        .workspace = std::move(wire_workspace),
                        .max_rows_per_participant =
                            transport_row_capacity_size,
                        .max_entries_per_participant =
                            transport_entry_capacity,
                        .d_model = d_model,
                        .top_k = top_k,
                        .tier_index = tier_index,
                        .domain_ordinal = routed_domain_ordinal,
                        .channel_identity = key,
                        .transaction_slot_count = 4096,
                        .transaction_topology =
                            transaction_topology_identity_,
                        .source_endpoint = continuation_endpoint,
                        .target_tier_priority =
                            config_.placement_plan
                                ->routed_tiers[static_cast<size_t>(tier_index)]
                                .priority,
                        .activation_graph_families =
                            activation_graph_families,
                        .local_lanes = std::move(local_lanes),
                        .enable_gpu_prefill_fp16_wire = gpu_to_gpu_prefill_fp16,
                    });
            rank_batch_transports_.emplace(key, transport);
            return transport;
        };

        /*
         * A same-node rank pair does not need a host-scheduled sparse graph.
         * Materialize one complete retained follower parent per physical GPU
         * and per admitted row geometry. Every endpoint graph owns the exact
         * dispatch-consume -> expert-compute -> return-pack sequence for all
         * source layers. The transaction executor launches the whole endpoint
         * set before observing any terminal event, so independent CUDA/ROCm
         * streams overlap instead of degenerating into submit/wait pairs.
         */
        const bool node_local_activation_epoch =
            config_.ownership == MoEOverlayParticipantGraphOwnership::RankLocalAuxiliary ||
            resolveMoEOverlayRankBatchTransportKind(
                *config_.mpi_context,
                source_world_rank,
                local_world_rank) ==
            MoEOverlayRankBatchTransportKind::NodeLocalSharedRows;
        if (node_local_activation_epoch)
        {
            struct MappedParticipantBinding
            {
                const MoEExpertOwnerParticipant *participant = nullptr;
                const RoutedExpertTier *tier = nullptr;
                int tier_index = -1;
                int routed_domain_ordinal = -1;
                std::shared_ptr<IMoEOverlayRankBatchTransport> transport;
                IMoEOverlayMappedActivationTransport *mapped = nullptr;
                std::shared_ptr<CachedParticipantGraph::MappedLaneAuthority>
                    authority;
            };

            const size_t graph_family_ordinal =
                spec.role == ParticipantGraphFamilyRole::Main
                    ? 0u
                    : static_cast<size_t>(spec.mtp_graph_depth + 1);
            std::vector<MappedParticipantBinding> bindings;
            bindings.reserve(local_participants_.size());

            for (size_t tier_index = 0;
                 tier_index < config_.placement_plan->routed_tiers.size();
                 ++tier_index)
            {
                const auto &tier =
                    config_.placement_plan->routed_tiers[tier_index];
                const int routed_domain_ordinal =
                    domain_ordinal(tier.domain);
                std::vector<int> participants;
                for (const auto *participant : local_participants_)
                {
                    if (!participant ||
                        participant->tier_idx !=
                            static_cast<int>(tier_index) ||
                        participant->domain_name != tier.domain ||
                        participant->world_rank != local_world_rank ||
                        (config_.ownership == MoEOverlayParticipantGraphOwnership::RankLocalAuxiliary &&
                         !std::binary_search(admitted_rank_local_endpoint_ids_.begin(),
                                             admitted_rank_local_endpoint_ids_.end(),
                                             participant->participant_id)))
                    {
                        continue;
                    }
                    participants.push_back(participant->participant_id);
                }
                if (participants.empty())
                    continue;
                std::sort(participants.begin(), participants.end());

                auto transport = transport_for_group(
                    static_cast<int>(tier_index),
                    routed_domain_ordinal,
                    participants);
                auto *const mapped = dynamic_cast<
                    IMoEOverlayMappedActivationTransport *>(transport.get());
                if (!mapped || graph_family_ordinal >=
                                   mapped->activationGraphFamilyCount())
                {
                    throw std::runtime_error(
                        "Node-local ExpertOverlay participant transport did not expose its required mapped activation family");
                }

                for (const int participant_id : participants)
                {
                    const auto *const participant =
                        owner_map_->participantForId(participant_id);
                    if (!participant || !participant->device.is_valid())
                    {
                        throw std::runtime_error(
                            "Node-local mapped ExpertOverlay resolved an invalid participant endpoint");
                    }
                    auto authority = std::make_shared<
                        CachedParticipantGraph::MappedLaneAuthority>();
                    authority->transport_lifetime = transport;
                    authority->participant_id = participant_id;
                    authority->tier_index = static_cast<int>(tier_index);
                    authority->tier_priority = tier.priority;
                    authority->routed_domain_ordinal =
                        routed_domain_ordinal;
                    authority->device = participant->device;
                    authority->graph_family_ordinal = graph_family_ordinal;
                    authority->protocol =
                        std::make_unique<MoEOverlayActivationEpochProtocol>(
                            mapped->activationEpochControl(
                                participant_id, graph_family_ordinal),
                            mapped->activationEpochConfig(
                                participant_id, graph_family_ordinal));
                    result->mapped_lane_authorities.push_back(authority);
                    bindings.push_back(MappedParticipantBinding{
                        .participant = participant,
                        .tier = &tier,
                        .tier_index = static_cast<int>(tier_index),
                        .routed_domain_ordinal = routed_domain_ordinal,
                        .transport = transport,
                        .mapped = mapped,
                        .authority = std::move(authority),
                    });
                }
            }
            if (bindings.empty() &&
                config_.ownership == MoEOverlayParticipantGraphOwnership::RankLocalAuxiliary &&
                admitted_rank_local_endpoint_ids_.empty())
            {
                // The physical controller/storage roster remains owned. There
                // is no expert-serving program or ticket endpoint to publish.
                result->admitted_empty = true;
                result->graph.reset();
                return result;
            }
            if (bindings.empty())
            {
                throw std::runtime_error(
                    "Node-local ExpertOverlay participant graph resolved no mapped endpoints");
            }

            const auto first_arena =
                serialCompactBufferArenaForParticipant(
                    *bindings.front().participant);
            // Storage capacities remain the original admitted families. Exact
            // live execution identities are independently retained through the
            // declared decode bound, including dynamic widths such as M3.
            std::vector<std::size_t> storage_rows;
            for(const auto& family:first_arena->families())storage_rows.push_back(family.row_capacity);
            // Learned drafts execute one token per request. Verifier widths
            // and shifted KV-only prefills belong to the continuation family;
            // neither declares a routed expert invocation for this parent.
            const bool draft_family=spec.role==ParticipantGraphFamilyRole::MTPDraft;
            const auto row_geometry=draft_family
                ?MoEOverlayFollowerRowGeometry::declareDraft(config_.max_request_count,storage_rows)
                :MoEOverlayFollowerRowGeometry::declare(row_capacity,config_.max_decode_activation_rows,
                    config_.prefill_graph_row_shapes,storage_rows);
            const auto& row_shapes=row_geometry.shapes();

            /*
             * CPU endpoints consume the same shared-page lanes as GPU parents,
             * but remain explicit host boundaries because no native GPU graph
             * can own CPU compute. Build them once at the admitted maximum and
             * reuse their stable packet views for every exact ticket shape. GPU
             * siblings are still submitted before this host loop starts.
             */
            for (const auto &binding : bindings)
            {
                if (!binding.participant->device.is_cpu())
                    continue;

                auto endpoint = std::make_unique<
                    CachedParticipantGraph::MappedCPUFollowerEndpoint>();
                endpoint->participant_id =
                    binding.participant->participant_id;
                endpoint->device = binding.participant->device;
                endpoint->tier_index = binding.tier_index;
                endpoint->routed_domain_ordinal =
                    binding.routed_domain_ordinal;
                endpoint->lane_authority = binding.authority;
                endpoint->lane = binding.mapped->activationDeviceLane(
                    endpoint->participant_id,
                    graph_family_ordinal,
                    endpoint->device);
                endpoint->input_rows =
                    std::make_shared<MoEOverlaySparseRows>(
                        endpoint->lane.hostDispatchPayload(row_capacity));
                endpoint->output_rows =
                    std::make_shared<MoEOverlayReturnRows>(
                        binding.transport->sharedReturnRows(
                            endpoint->participant_id));
                endpoint->layers.reserve(spec.source_layers.size());

                const bool joint=config_.participant_residency->isJointParticipant(endpoint->participant_id);
                const auto participant_residency = joint?std::shared_ptr<MoEOverlayParticipantResidency>{}:
                    config_.participant_residency->endpoint(endpoint->participant_id);
                if (!joint && !participant_residency)
                {
                    throw std::runtime_error(
                        "Mapped ExpertOverlay CPU endpoint is missing its prepared residency authority");
                }
                const auto compact_arena =
                    serialCompactBufferArenaForParticipant(
                        *binding.participant);

                for (const int layer : spec.source_layers)
                {
                    const auto expert_mask =
                        owner_map_->expertMaskForParticipant(
                            layer,
                            endpoint->participant_id,
                            num_experts);
                    std::vector<MoEOverlayPreparedExpertPayload>
                        prepared_payloads;
                    std::string residency_error;
                    std::shared_ptr<const MoEOverlayParticipantResidencyRegistry::JointLayer> joint_layer;
                    if(joint) {
                        if(!config_.cpu_joint_context || !prepared_store_)
                            throw std::runtime_error("CPU joint follower has no original prepared source/domain");
                        joint_layer=adoptMoEOverlayJointGraphDeclaration(*config_.participant_residency,registry,
                            prepared_store_->modelId(),endpoint->participant_id,layer,num_experts,d_model,expert_intermediate);
                        if(!joint_layer || !joint_layer->sourceSelection() ||
                           !compact_arena->cpuJointLayout() || !compact_arena->cpuJointPackedRecords())
                            throw std::runtime_error("CPU joint follower lost original fragment publication or serial roots");

                    }
                    if (!joint && (!resolveMoEOverlayPreparedExpertPayloads(
                            registry,
                            *binding.participant,
                            layer,
                            num_experts,
                            expert_mask,
                            prepared_payloads,
                            &residency_error) ||
                        !config_.participant_residency
                             ->registerInitialLayer(
                                 endpoint->participant_id,
                                 layer,
                                 expert_mask,
                                 prepared_payloads,
                                 &residency_error)))
                    {
                        throw std::runtime_error(
                            "Mapped ExpertOverlay CPU follower could not publish its initial residency bank for layer " +
                            std::to_string(layer) + " participant " +
                            std::to_string(endpoint->participant_id) +
                            ": " + residency_error);
                    }

                    MoELocalExpertStage::Params local_params;
                    local_params.device_id = endpoint->device;
                    local_params.input_rows_lifetime = endpoint->input_rows;
                    local_params.output_rows_lifetime = endpoint->output_rows;
                    if(!joint)local_params.cpu_canonical_route_return =
                        MoELocalExpertStage::CPUCanonicalRouteReturnBinding{
                            .original_route_slots =
                                endpoint->lane.dispatch.original_route_slots,
                            .compact_route_slots =
                                endpoint->lane.dispatch.compact_route_slots,
                            .preweighted_route_contributions =
                                endpoint->lane.returned
                                    .canonical_route_contributions_fp32,
                            .route_slot_capacity =
                                endpoint->lane.returned.route_slot_capacity,
                        };
                    if(joint) {
                        local_params.cpu_joint_layer=joint_layer;
                        local_params.cpu_joint_route_producer=MoELocalExpertStage::CPUJointRouteProducerBinding{
                            .packed_rows=compact_arena->cpuJointPackedRecords(),
                            .original_route_slots=endpoint->lane.dispatch.original_route_slots,
                            .compact_route_slots=endpoint->lane.dispatch.compact_route_slots,
                            .route_slot_capacity=endpoint->lane.returned.route_slot_capacity};
                    }
                    local_params.serial_compact_buffer_arena = compact_arena;
                    local_params.graph_row_capacity =
                        static_cast<size_t>(row_capacity);
                    local_params.completion_policy =
                        MoELocalExpertStage::CompletionPolicy::Inline;
                    local_params.num_experts = num_experts;
                    local_params.top_k = top_k;
                    local_params.d_model = d_model;
                    local_params.expert_intermediate = joint?joint_layer->sourceSelection()->ownership()
                        .projection(WeightRole::MoEExpertUp).rows:expert_intermediate;
                    local_params.layer_idx = layer;
                    local_params.cpu_workspace_source = cpuExpertWorkspaceSource(
                        config_.model_context->concreteLoader(), layer, d_model, expert_intermediate, num_experts);
                    local_params.expert_mask = expert_mask;
                    local_params.overlay_participant_residency =
                        participant_residency;
                    local_params.prepared_store = prepared_store_.get();
                    local_params.physical_memory_authority = prepared_store_
                        ? prepared_store_->physicalMemoryAuthority() : nullptr;
                    local_params.expert_registry = &registry;
                    local_params.runtime_participant_index =
                        endpoint->participant_id;
                    local_params.expert_weight_resolution_policy =
                        MoELocalExpertStage::
                            ExpertWeightResolutionPolicy::RegistryOnly;

                    if(joint) {
                        local_params.prepared_gate_gemm.assign(num_experts,nullptr);
                        local_params.prepared_up_gemm.assign(num_experts,nullptr);
                        local_params.prepared_down_gemm.assign(num_experts,nullptr);
                        local_params.expert_mask.assign(num_experts,false);
                    }
                    if(!joint)(void)registry.populateExpertEnginesForParticipant(
                        binding.tier->domain,
                        endpoint->device,
                        binding.participant->world_rank,
                        binding.participant->domain_participant_index,
                        layer,
                        num_experts,
                        local_params.prepared_gate_gemm,
                        local_params.prepared_up_gemm,
                        local_params.prepared_down_gemm);
                    /*
                     * An initially empty tier deliberately has no engines yet.
                     * The epoch-indexed residency bank can install arrivals later,
                     * and the stage resolves that exact bank on every invocation.
                     */
                    if (!joint && hasActiveMask(expert_mask) &&
                        !MoELocalExpertStage::prepareExpertGemmEngines(
                            local_params))
                    {
                        throw std::runtime_error(
                            "Mapped ExpertOverlay CPU follower has incomplete prepared expert engines for layer " +
                            std::to_string(layer) + " participant " +
                            std::to_string(endpoint->participant_id));
                    }

                    std::unique_ptr<MoECPUExpertShardRouteStage> joint_collective;
                    if(joint) {
                        const auto &geometry=*compact_arena->cpuJointLayout();
                        joint_collective=std::make_unique<MoECPUExpertShardRouteStage>(MoECPUExpertShardRouteStage::Params{
                            .original_source=joint_layer->sourceSelection(),
                            .tp=config_.cpu_joint_context.get(),.packed_records=compact_arena->cpuJointPackedRecords().get(),
                            .combined_routes=geometry.isRoot()?compact_arena->cpuCanonicalRoutes().get():nullptr,
                            .source_ranges=compact_arena->cpuJointSourceRanges(),
                            .participant_route_maps=compact_arena->cpuJointRouteMaps(),
                            .required_route_weights=compact_arena->cpuJointRequiredRoutes(),
                            .rows=row_capacity,.top_k=top_k,.columns=d_model,.root_participant=static_cast<int>(geometry.rootMember()),
                            .stage_name="cpu_joint_original_routes.layer"+std::to_string(layer),
                            .original_return_rows=endpoint->output_rows.get(),
                            .original_preweighted_routes=geometry.isRoot()?endpoint->lane.returned.canonical_route_contributions_fp32:nullptr,
                            .original_return_route_capacity=endpoint->lane.returned.route_slot_capacity,
                            .original_sparse_routes=endpoint->input_rows.get(),.original_prepared_layer=joint_layer,
                            .original_tp_owner=config_.cpu_joint_context});
                    }
                    endpoint->layers.push_back(
                        CachedParticipantGraph::MappedCPUFollowerLayer{
                            .model_layer_index = layer,
                            .stage_ordinal =
                                binding.mapped->activationStageOrdinal(
                                    graph_family_ordinal, layer),
                            .local_expert =
                                std::make_unique<MoELocalExpertStage>(
                                    std::move(local_params)),
                            .joint_route_collective=std::move(joint_collective),
                        });
                }
                result->mapped_cpu_endpoints.push_back(std::move(endpoint));
            }

            if (!result->mapped_cpu_endpoints.empty())
            {
                // CPU followers are explicit heterogeneous boundaries, not
                // nodes in a GPU parent. Register them as non-graph consumers
                // of the same serial-family allocator; otherwise a graph-only
                // walk silently omits every CPU layer's invocation workspace.
                std::vector<WorkspaceConsumerRequest> consumers;
                for (const auto &endpoint : result->mapped_cpu_endpoints)
                    for (const auto &layer : endpoint->layers)
                        consumers.push_back({.consumer = layer.local_expert.get(),
                            .device = endpoint->device, .m = row_capacity,
                            .shape_policy = WorkspaceConsumerShapePolicy::FixedDeclaredShape});
                result->workspace_allocator = serial_graph_family_workspace_allocator_;
                WorkspaceSizingHints hints;
                hints.max_seq_len = row_capacity;
                hints.serial_family_max_rows = row_capacity;
                hints.d_model = d_model;
                hints.batch_size = 1;
                hints.graph_family_policy = WorkspaceGraphFamilyPolicy::SerialDeviceFamilyExactParticipant;
                if (!result->workspace_allocator->allocateForGraph(*result->graph, hints, consumers))
                    throw std::runtime_error("Mapped CPU followers could not bind their admitted serial workspace");
            }

            /*
             * Materialize the largest geometry first. The runner-owned
             * WorkspaceAllocator spans main, verifier, and MTP families as
             * well as every physical-row specialization. It therefore binds
             * every smaller retained graph to the already-published maximum
             * buffers and rejects any shape the setup BOM omitted. Iterating
             * in ascending order would ask a live serial allocator to grow
             * after its first graph captured addresses, which is intentionally
             * forbidden.
             */
            for (auto shape_it = row_shapes.rbegin();
                 shape_it != row_shapes.rend();
                 ++shape_it)
            {
                const int physical_rows = shape_it->physical_rows;
                CachedParticipantGraph::MappedGPUFollowerShape shape;
                shape.physical_rows = physical_rows;
                shape.endpoints.reserve(bindings.size());

                for (const auto &binding : bindings)
                {
                    if (!binding.participant->device.is_gpu())
                        continue;
                    const DeviceId target_device =
                        binding.participant->device;
                    auto endpoint = std::make_shared<
                        CachedParticipantGraph::MappedGPUFollowerEndpoint>();
                    endpoint->participant_id =
                        binding.participant->participant_id;
                    endpoint->device = target_device;
                    endpoint->physical_rows = physical_rows;
                    endpoint->lane_authority = binding.authority;
                    endpoint->graph = std::make_unique<ComputeGraph>();
                    endpoint->graph->setNativeCaptureEnvelope(
                        GraphNativeCaptureEnvelope::
                            DeviceOwnedTimelineTransaction);
                    endpoint->arena = std::make_unique<BufferArena>();
                    endpoint->active_rows =
                        std::make_shared<INT32Tensor>(
                            std::vector<size_t>{1u});

                    const auto compact_arena =
                        serialCompactBufferArenaForParticipant(
                            *binding.participant);
                    const auto *const tensor_family =
                        compact_arena->smallestFamilySupporting(
                            static_cast<size_t>(physical_rows));
                    if (!tensor_family ||
                        tensor_family->row_capacity != shape_it->backing_rows ||
                        tensor_family->row_capacity < static_cast<size_t>(physical_rows) ||
                        !tensor_family->hidden ||
                        !tensor_family->routing_indices ||
                        !tensor_family->routing_weights ||
                        !tensor_family->output)
                    {
                        throw std::logic_error(
                            "Mapped ExpertOverlay follower exact rows do not fit their original admitted compact backing");
                    }
                    auto canonical_it =
                        participant_canonical_route_buffers_.find(
                            endpoint->participant_id);
                    if (canonical_it ==
                        participant_canonical_route_buffers_.end())
                    {
                        /* Only the node-local mapped GPU path consumes this
                         * bank. Allocate it on first graph-family materialization
                         * at the shared transport maximum, then reuse that stable
                         * address for main, verifier, and every MTP family whose
                         * executions are scheduler-serialized. */
                        auto canonical_routes =
                            std::make_shared<FP32Tensor>(
                                std::vector<size_t>{
                                    transport_entry_capacity,
                                    static_cast<size_t>(d_model)});
                        canonical_it =
                            participant_canonical_route_buffers_
                                .emplace(
                                    endpoint->participant_id,
                                    std::move(canonical_routes))
                                .first;
                    }
                    if (canonical_it ==
                            participant_canonical_route_buffers_.end() ||
                        !canonical_it->second ||
                        canonical_it->second->numel() !=
                            checkedGeometryProduct(
                                transport_entry_capacity,
                                static_cast<size_t>(d_model),
                                "participant canonical route elements"))
                    {
                        throw std::logic_error(
                            "Mapped ExpertOverlay follower has no participant-owned canonical route bank");
                    }
                    const auto &canonical_routes = canonical_it->second;

                    TransferEngine::allocateDeviceStorage(
                        tensor_family->hidden.get(), target_device);
                    TransferEngine::allocateDeviceStorage(
                        tensor_family->routing_indices.get(), target_device);
                    TransferEngine::allocateDeviceStorage(
                        tensor_family->routing_weights.get(), target_device);
                    TransferEngine::allocateDeviceStorage(
                        tensor_family->output.get(), target_device);
                    TransferEngine::allocateDeviceStorage(
                        canonical_routes.get(), target_device);
                    TransferEngine::allocateDeviceStorage(
                        endpoint->active_rows.get(), target_device);
                    if (!endpoint->active_rows->gpu_data_ptr())
                    {
                        throw std::runtime_error(
                            "Mapped ExpertOverlay follower active-row scalar has no device storage");
                    }

                    if (!endpoint->arena->registerExternalBuffer(
                            BufferId::NORMALIZED,
                            tensor_family->hidden.get()) ||
                        !endpoint->arena->registerExternalBuffer(
                            BufferId::MOE_EXPERT_INDICES,
                            tensor_family->routing_indices.get()) ||
                        !endpoint->arena->registerExternalBuffer(
                            BufferId::MOE_EXPERT_WEIGHTS,
                            tensor_family->routing_weights.get()) ||
                        !endpoint->arena->registerExternalBuffer(
                            BufferId::MOE_COMBINED_OUTPUT,
                            tensor_family->output.get()) ||
                        !endpoint->arena->registerExternalBuffer(
                            BufferId::MOE_CANONICAL_ROUTE_CONTRIBUTIONS,
                            canonical_routes.get()))
                    {
                        throw std::runtime_error(
                            "Mapped ExpertOverlay follower could not register its complete arena frontier");
                    }

                    endpoint->tensor_lifetimes = {
                        tensor_family->hidden,
                        tensor_family->routing_indices,
                        tensor_family->routing_weights,
                        tensor_family->output,
                        canonical_routes,
                    };

                    auto lane = binding.mapped->activationDeviceLane(
                        endpoint->participant_id,
                        graph_family_ordinal,
                        target_device);
                    if (!lane.valid())
                    {
                        throw std::runtime_error(
                            "Mapped ExpertOverlay follower received an invalid device lane");
                    }
                    endpoint->lane = lane;

                    ParticipantGpuRuntime *gpu_runtime =
                        &participantGpuRuntimeForParticipant(
                            endpoint->participant_id);
                    std::string epoch_acquire_name;
                    if (gpu_runtime->device != target_device ||
                        !gpu_runtime->runtime_table ||
                        !gpu_runtime->epoch_arena)
                    {
                        throw std::logic_error(
                            "Mapped follower endpoint disagrees with its durable GPU runtime authority");
                    }

                    epoch_acquire_name =
                        "mapped_follower_p" +
                        std::to_string(endpoint->participant_id) +
                        "_epoch_acquire";
                    MoEOverlayEpochBoundaryStage::Params acquire_params;
                    acquire_params.device_id = target_device;
                    acquire_params.arena = gpu_runtime->epoch_arena;
                    acquire_params.request_slot = 0u;
                    acquire_params.operation =
                        MoEOverlayEpochBoundaryStage::Operation::Acquire;
                    if (spec.source_layers.empty())
                    {
                        throw std::logic_error(
                            "Mapped follower epoch transaction has no routed layer");
                    }
                    const std::uint32_t first_stage_ordinal =
                        binding.mapped->activationStageOrdinal(
                            graph_family_ordinal,
                            spec.source_layers.front());
                    if (first_stage_ordinal != 0u)
                    {
                        throw std::logic_error(
                            "Mapped follower epoch transaction does not begin at stage zero");
                    }
                    acquire_params.peer_epoch_source =
                        MoEOverlayEpochBoundaryStage::PeerEpochSource{
                            .mapped_region = lane.mapped_region,
                            .binding = {
                                .control = lane.control_device,
                                .grant = lane.grant_device,
                                .stage_ordinal = first_stage_ordinal,
                            },
                        };
                    acquire_params.stage_name = epoch_acquire_name;
                    endpoint->graph->addNode(
                        epoch_acquire_name,
                        std::make_unique<MoEOverlayEpochBoundaryStage>(
                            std::move(acquire_params)),
                        target_device);
                    std::shared_ptr<TensorBase> rocm_follower_scratch;
#ifdef HAVE_ROCM
                    if (physical_rows == 1 && target_device.is_rocm() &&
                        (d_model % 32) == 0 && (expert_intermediate % 32) == 0 &&
                        top_k > 0 && top_k <= 16)
                    {
                        const std::size_t scratch_bytes =
                            rocm::MappedFollowerScratchLayout::totalBytes(
                                d_model, top_k, expert_intermediate);
                        const std::size_t scratch_words =
                            (scratch_bytes + sizeof(std::int32_t) - 1u) /
                            sizeof(std::int32_t);
                        auto scratch_it =
                            participant_mapped_follower_scratch_buffers_.find(
                                endpoint->participant_id);
                        if (scratch_it ==
                            participant_mapped_follower_scratch_buffers_.end())
                        {
                            auto buf = std::make_shared<INT32Tensor>(
                                std::vector<size_t>{scratch_words});
                            TransferEngine::allocateDeviceStorage(
                                buf.get(), target_device);
                            if (!buf->gpu_data_ptr())
                            {
                                throw std::runtime_error(
                                    "ROCm mapped follower scratch buffer has no device storage");
                            }
                            IBackend *const backend = getBackendFor(target_device);
                            void *const init_stream =
                                participantWorkerStream(target_device);
                            if (!backend || !init_stream)
                            {
                                throw std::runtime_error(
                                    "ROCm mapped follower scratch init lacks backend or stream");
                            }
                            gpu_runtime->worker->submitAndWait(
                                [&]
                                {
                                    if (!backend->memset(
                                            buf->gpu_data_ptr(),
                                            0,
                                            64u,
                                            target_device.gpu_ordinal(),
                                            init_stream))
                                    {
                                        throw std::runtime_error(
                                            "ROCm mapped follower could not zero resident barrier words");
                                    }
                                });
                            scratch_it =
                                participant_mapped_follower_scratch_buffers_
                                    .emplace(
                                        endpoint->participant_id,
                                        std::move(buf))
                                    .first;
                        }
                        rocm_follower_scratch = scratch_it->second;
                        endpoint->tensor_lifetimes.push_back(rocm_follower_scratch);
                    }
#endif
                    std::string previous_layer_return;
                    for (const int layer : spec.source_layers)
                    {
                        const std::uint32_t stage_ordinal =
                            binding.mapped->activationStageOrdinal(
                                graph_family_ordinal, layer);
                        const std::string stem =
                            "mapped_follower_p" +
                            std::to_string(endpoint->participant_id) +
                            "_layer" + std::to_string(layer);

                        const auto expert_mask =
                            owner_map_->expertMaskForParticipant(
                                layer,
                                endpoint->participant_id,
                                num_experts);
                        std::vector<MoEOverlayPreparedExpertPayload>
                            prepared_payloads;
                        std::string residency_error;
                        if (!resolveMoEOverlayPreparedExpertPayloads(
                                registry,
                                *binding.participant,
                                layer,
                                num_experts,
                                expert_mask,
                                prepared_payloads,
                                &residency_error) ||
                            !config_.participant_residency
                                 ->registerInitialLayer(
                                     endpoint->participant_id,
                                     layer,
                                     expert_mask,
                                     prepared_payloads,
                                     &residency_error))
                        {
                            throw std::runtime_error(
                                "Mapped ExpertOverlay follower could not publish its initial residency bank for layer " +
                                std::to_string(layer) + " participant " +
                                std::to_string(endpoint->participant_id) +
                                ": " + residency_error);
                        }
                        std::vector<ITensorGemm *> prepared_gate;
                        std::vector<ITensorGemm *> prepared_up;
                        std::vector<ITensorGemm *> prepared_down;
                        (void)registry.populateExpertEnginesForParticipant(
                            binding.tier->domain,
                            target_device,
                            binding.participant->world_rank,
                            binding.participant->domain_participant_index,
                            layer,
                            num_experts,
                            prepared_gate,
                            prepared_up,
                            prepared_down);
                        if (hasActiveMask(expert_mask))
                        {
                            MoELocalExpertStage::Params validation_params;
                            validation_params.device_id = target_device;
                            validation_params.num_experts = num_experts;
                            validation_params.top_k = top_k;
                            validation_params.d_model = d_model;
                            validation_params.expert_intermediate =
                                expert_intermediate;
                            validation_params.layer_idx = layer;
                            validation_params.expert_mask = expert_mask;
                            validation_params.prepared_gate_gemm =
                                prepared_gate;
                            validation_params.prepared_up_gemm = prepared_up;
                            validation_params.prepared_down_gemm =
                                prepared_down;
                            validation_params.expert_weight_resolution_policy =
                                MoELocalExpertStage::
                                    ExpertWeightResolutionPolicy::RegistryOnly;
                            if (!MoELocalExpertStage::
                                     prepareExpertGemmEngines(
                                         validation_params))
                            {
                                throw std::runtime_error(
                                    "Mapped ExpertOverlay follower has incomplete prepared expert engines for layer " +
                                    std::to_string(layer) + " participant " +
                                    std::to_string(endpoint->participant_id));
                            }
                        }

                        auto &initialized =
                            gpu_runtime->initialized_layers.at(
                                gpu_runtime->runtime_table->storageIndexForModelLayer(layer));
                        if (initialized == 0u)
                        {
                                    MoEPlacementUpdate update;
                                    update.epoch = 1u;
                                    update.expert_count =
                                        static_cast<std::uint32_t>(
                                            num_experts);
                                    update.participant_id =
                                        gpu_runtime->domain_participant_id;
                                    update.participant_count =
                                        gpu_runtime->domain_participant_count;
                                    update.experts.resize(
                                        static_cast<std::size_t>(
                                            num_experts));
                                    update.local_compute_mask.assign(
                                        static_cast<std::size_t>(
                                            num_experts),
                                        0u);
                                    update.replica_role.assign(
                                        static_cast<std::size_t>(
                                            num_experts),
                                        static_cast<std::uint8_t>(
                                            DeviceMoEReplicaRole::None));
                                    update.resident_participant_mask.assign(
                                        static_cast<std::size_t>(
                                            num_experts),
                                        0u);
                                    update.overlay_route_participant.assign(
                                        static_cast<std::size_t>(
                                            num_experts),
                                        -1);

                                    for (int expert = 0;
                                         expert < num_experts;
                                         ++expert)
                                    {
                                        const auto *placement=owner_map_->placementFor(layer,expert);
                                        if(placement && placement->shard_cohort) {
                                            if(placement->domain_name==binding.participant->domain_name)
                                                throw std::logic_error("GPU scalar runtime cannot represent its own joint CPU domain");
                                            auto &external=update.experts[static_cast<size_t>(expert)];
                                            external.logical_expert_id=expert;external.owner_participant=-1;
                                            update.overlay_route_participant[static_cast<size_t>(
                                                expert)] =
                                                MoEOverlayRouteTarget::cohort(
                                                    placement->tier_idx)
                                                    .word();
                                            continue; // Fixed CPU cohort remains external to GPU
                                                      // compute.
                                        }
                                        const auto *const owner =
                                            owner_map_->ownerFor(
                                                layer, expert);
                                        if (!owner ||
                                            owner->owner_participant < 0)
                                        {
                                            throw std::logic_error(
                                                "Mapped follower runtime initialization found an unowned expert");
                                        }

                                        auto &descriptor =
                                            update.experts[
                                                static_cast<std::size_t>(
                                                    expert)];
                                        descriptor.logical_expert_id = expert;
                                        update.overlay_route_participant[
                                            static_cast<std::size_t>(expert)] =
                                            owner->owner_participant;
                                        if (owner->tier_idx ==
                                            binding.participant->tier_idx)
                                        {
                                            if (owner
                                                        ->domain_participant_index <
                                                    0 ||
                                                static_cast<std::uint32_t>(
                                                    owner
                                                        ->domain_participant_index) >=
                                                    update.participant_count)
                                            {
                                                throw std::logic_error(
                                                    "Mapped follower owner has no valid domain-local runtime identity");
                                            }
                                            descriptor.owner_participant =
                                                owner
                                                    ->domain_participant_index;
                                            update
                                                .resident_participant_mask[
                                                    static_cast<std::size_t>(
                                                        expert)] =
                                                1u << static_cast<std::uint32_t>(
                                                    owner
                                                        ->domain_participant_index);
                                        }
                                        else
                                        {
                                            descriptor.owner_participant = -1;
                                        }

                                        if (owner->owner_participant !=
                                            endpoint->participant_id)
                                        {
                                            continue;
                                        }
                                        if (!expert_mask[
                                                static_cast<std::size_t>(
                                                    expert)] ||
                                            !exportDeviceMoEExpertWeightDescriptors(
                                                prepared_gate[
                                                    static_cast<std::size_t>(
                                                        expert)],
                                                prepared_up[
                                                    static_cast<std::size_t>(
                                                        expert)],
                                                prepared_down[
                                                    static_cast<std::size_t>(
                                                        expert)],
                                                d_model,
                                                expert_intermediate,
                                                descriptor))
                                        {
                                            throw std::runtime_error(
                                                "Mapped follower could not export the authoritative prepared expert descriptor");
                                        }
                                        descriptor.logical_expert_id = expert;
                                        descriptor.owner_participant =
                                            static_cast<std::int32_t>(
                                                update.participant_id);
                                        descriptor.local_slot = expert;
                                        descriptor.flags = toMoEExpertFlags(
                                            DeviceMoEExpertFlags::Valid |
                                            DeviceMoEExpertFlags::Resident |
                                            DeviceMoEExpertFlags::LocalCompute |
                                            DeviceMoEExpertFlags::PreferredOwner);
                                        update.local_compute_mask[
                                            static_cast<std::size_t>(expert)] =
                                            1u;
                                        update.replica_role[
                                            static_cast<std::size_t>(expert)] =
                                            static_cast<std::uint8_t>(
                                                DeviceMoEReplicaRole::Primary);
                                        update
                                            .resident_participant_mask[
                                                static_cast<std::size_t>(
                                                    expert)] |=
                                            1u << update.participant_id;
                                    }

                                    void *const publication_stream =
                                        participantWorkerStream(
                                            target_device);
                                    gpu_runtime->worker->submitAndWait(
                                        [&]
                                        {
                                            gpu_runtime
                                                ->runtime_table
                                                ->prepareInactiveBank(
                                                    layer, update);
                                            gpu_runtime
                                                ->runtime_table
                                                ->flipActiveBank(
                                                    layer,
                                                    update.epoch,
                                                    publication_stream);
                                        });
                            initialized = 1u;
                        }

#ifdef HAVE_ROCM
                        if (rocm_follower_scratch && hasActiveMask(expert_mask))
                        {
                            const auto single_payload = lane.dispatchPayload(1);
                            if (single_payload.valid() &&
                                single_payload.selection.physical_rows == 1 &&
                                single_payload.selection.usesCompactRows())
                            {
                                bool all_native = true;
                                std::uint32_t gate_mask = 0u;
                                std::uint32_t down_mask = 0u;
                                for (int e = 0; e < num_experts && all_native; ++e)
                                {
                                    if (!expert_mask[static_cast<std::size_t>(e)])
                                        continue;
                                    DeviceMoEExpertDescriptor desc{};
                                    if (!exportDeviceMoEExpertWeightDescriptors(
                                            prepared_gate[static_cast<std::size_t>(e)],
                                            prepared_up[static_cast<std::size_t>(e)],
                                            prepared_down[static_cast<std::size_t>(e)],
                                            d_model, expert_intermediate, desc) ||
                                        desc.weight_format != DeviceMoEWeightFormat::NativeQuantised)
                                    {
                                        all_native = false;
                                        break;
                                    }
                                    gate_mask |= (std::uint32_t{1u} << desc.gate.codebook_id) |
                                                 (std::uint32_t{1u} << desc.up.codebook_id);
                                    down_mask |= (std::uint32_t{1u} << desc.down.codebook_id);
                                }
                                if (all_native &&
                                    rocm::fastMappedFollowerMoEKernelFunction(gate_mask, down_mask) != nullptr)
                                {
                                    const std::string fused_name =
                                        stem + "_fused_mapped_follower";
                                    MoEOverlayROCmMappedFollowerStage::Params fused_params;
                                    fused_params.device_id = target_device;
                                    fused_params.lane = lane;
                                    fused_params.placement =
                                        gpu_runtime->runtime_table->overlayRoutePlacementBinding(layer);
                                    fused_params.runtime_layer =
                                        gpu_runtime->runtime_table->deviceLayerState(layer);
                                    fused_params.service_telemetry =
                                        gpu_runtime->runtime_table->deviceOverlayServiceTelemetryBinding(layer);
                                    fused_params.service_graph_role =
                                        &lane.grant_device->graph_role;
                                    fused_params.active_row_count_device =
                                        static_cast<std::int32_t *>(endpoint->active_rows->gpu_data_ptr());
                                    fused_params.scratch_buffer = rocm_follower_scratch.get();
                                    fused_params.hidden = tensor_family->hidden.get();
                                    fused_params.routing_indices = tensor_family->routing_indices.get();
                                    fused_params.routing_weights = tensor_family->routing_weights.get();
                                    fused_params.local_canonical_route_contributions = canonical_routes.get();
                                    fused_params.d_model = d_model;
                                    fused_params.expert_intermediate = expert_intermediate;
                                    fused_params.num_experts = num_experts;
                                    fused_params.top_k = top_k;
                                    fused_params.model_layer_index = layer;
                                    fused_params.stage_ordinal = stage_ordinal;
                                    fused_params.gate_execution_mask = gate_mask;
                                    fused_params.down_execution_mask = down_mask;
                                    fused_params.observe_histogram = true;
                                    fused_params.stage_name = fused_name;
                                    endpoint->graph->addNode(
                                        fused_name,
                                        std::make_unique<MoEOverlayROCmMappedFollowerStage>(
                                            std::move(fused_params)),
                                        target_device);
                                    if (!previous_layer_return.empty())
                                    {
                                        endpoint->graph->addDependency(
                                            fused_name, previous_layer_return);
                                    }
                                    else if (!epoch_acquire_name.empty())
                                    {
                                        endpoint->graph->addDependency(
                                            fused_name, epoch_acquire_name);
                                    }
                                    previous_layer_return = fused_name;
                                    continue;
                                }
                            }
                        }
#endif

                        const std::string consume_name =
                            stem + "_dispatch_consume";
                        MoEOverlayActivationDispatchConsumeStage::Params
                            consume_params;
                        consume_params.device_id = target_device;
                        consume_params.lane = lane;
                        consume_params.placement =
                            gpu_runtime->runtime_table
                                ->overlayRoutePlacementBinding(layer);
                        consume_params.hidden = tensor_family->hidden.get();
                        consume_params.routing_indices =
                            tensor_family->routing_indices.get();
                        consume_params.routing_weights =
                            tensor_family->routing_weights.get();
                        consume_params.active_row_count_device =
                            static_cast<std::int32_t *>(
                                endpoint->active_rows->gpu_data_ptr());
                        consume_params.physical_rows = physical_rows;
                        consume_params.stage_ordinal = stage_ordinal;
                        consume_params.model_layer_index = layer;
                        endpoint->graph->addNode(
                            consume_name,
                            ComputeStageFactory::
                                createMoEOverlayActivationDispatchConsume(
                                    consume_params),
                            target_device);
                        if (!previous_layer_return.empty())
                        {
                            endpoint->graph->addDependency(
                                consume_name, previous_layer_return);
                        }
                        else if (!epoch_acquire_name.empty())
                        {
                            endpoint->graph->addDependency(
                                consume_name, epoch_acquire_name);
                        }
                        std::string compute_dependency = consume_name;

                        if (hasActiveMask(expert_mask))
                        {
                            MoEExpertComputeStage::Params compute_params;
                            compute_params.device_id = target_device;
                            compute_params.input = tensor_family->hidden.get();
                            compute_params.seq_len = physical_rows;
                            compute_params.d_model = d_model;
                            compute_params.num_experts = num_experts;
                            compute_params.top_k = top_k;
                            compute_params.expert_intermediate =
                                expert_intermediate;
                            compute_params.layer_idx = layer;
                            compute_params.expert_mask = expert_mask;
                            compute_params.routing_indices =
                                tensor_family->routing_indices.get();
                            compute_params.routing_weights =
                                tensor_family->routing_weights.get();
                            compute_params.output =
                                tensor_family->output.get();
                            compute_params.canonical_route_contributions =
                                canonical_routes.get();
                            compute_params.canonical_route_arithmetic =
                                MoECanonicalRouteArithmeticPolicy::
                                    PreweightedContributionThenOrderedAdd;
                            compute_params.canonical_route_layout =
                                MoECanonicalRoutePublicationLayout::
                                    DenseOriginalRouteSlots;
                            compute_params.active_row_count_device =
                                static_cast<const std::int32_t *>(
                                    endpoint->active_rows->gpu_data_ptr());
                            /*
                             * A one-row follower consumes the packet's
                             * device-owned router tensors through the fused
                             * explicit-routing decode path. It is ordinary
                             * decode, not an MTP grouped-prefill verifier, and
                             * must retain the economical M=1 kernel route.
                             */
                            compute_params
                                .require_device_routing_tensor_decode = true;
                            compute_params.output_registered_in_arena = true;
                            compute_params.prepared_gate_gemm =
                                std::move(prepared_gate);
                            compute_params.prepared_up_gemm =
                                std::move(prepared_up);
                            compute_params.prepared_down_gemm =
                                std::move(prepared_down);
                            compute_params.prepared_store =
                                prepared_store_.get();
                            compute_params.expert_registry = &registry;
                            compute_params.moe_runtime_table =
                                gpu_runtime->runtime_table.get();
                            compute_params
                                .runtime_service_graph_role_device =
                                &lane.grant_device->graph_role;
                            compute_params.use_runtime_row_grouping = true;
                            compute_params.weight_descriptor_source =
                                MoEDecodeDescriptorSource::
                                    RuntimePlacementTable;
                            compute_params
                                .runtime_decode_has_explicit_owner_metadata =
                                true;
                            compute_params.my_socket_id =
                                static_cast<int>(
                                    gpu_runtime->domain_participant_id);
                            compute_params.participant_count =
                                static_cast<int>(
                                    gpu_runtime->domain_participant_count);
                            compute_params.routed_assignment_policy =
                                RoutedExpertAssignmentPolicy::StaticOwner;
                            compute_params.routed_row_execution_policy =
                                RoutedExpertRowExecutionPolicy::
                                    ParticipantAssigned;

                            auto compute_stage =
                                std::make_unique<MoEExpertComputeStage>(
                                    std::move(compute_params));
                            compute_stage->releaseRawExpertWeights();
                            const std::string compute_name =
                                stem + "_expert_compute";
                            endpoint->graph->addNode(
                                compute_name,
                                std::move(compute_stage),
                                target_device);
                            endpoint->graph->addDependency(
                                compute_name, consume_name);
                            compute_dependency = compute_name;
                        }

                        MoEOverlayActivationReturnPackStage::Params
                            return_params;
                        return_params.device_id = target_device;
                        return_params.lane = lane;
                        return_params.local_canonical_route_contributions =
                            canonical_routes.get();
                        return_params.physical_rows = physical_rows;
                        return_params.stage_ordinal = stage_ordinal;
                        return_params.model_layer_index = layer;
                        const std::string return_name =
                            stem + "_return_pack";
                        endpoint->graph->addNode(
                            return_name,
                            ComputeStageFactory::
                                createMoEOverlayActivationReturnPack(
                                    return_params),
                            target_device);
                        endpoint->graph->addDependency(
                            return_name, compute_dependency);
                        previous_layer_return = return_name;
                    }

                    if (previous_layer_return.empty())
                    {
                        throw std::logic_error(
                            "Mapped follower epoch transaction contains no routed layer to release");
                    }
                    const std::string release_name =
                        "mapped_follower_p" +
                        std::to_string(endpoint->participant_id) +
                        "_epoch_release";
                    MoEOverlayEpochBoundaryStage::Params release_params;
                    release_params.device_id = target_device;
                    release_params.arena = gpu_runtime->epoch_arena;
                    release_params.request_slot = 0u;
                    release_params.operation =
                        MoEOverlayEpochBoundaryStage::Operation::Release;
                    if (config_.durable_maintenance_policy ==
                            MoEOverlayDurableMaintenancePolicy::
                                DynamicPlacement &&
                        gpu_runtime->controller_binding)
                    {
                        release_params.retirement_readiness_controller =
                            *gpu_runtime->controller_binding;
                    }
                    release_params.stage_name = release_name;
                    endpoint->graph->addNode(
                        release_name,
                        std::make_unique<MoEOverlayEpochBoundaryStage>(
                            std::move(release_params)),
                        target_device);
                    endpoint->graph->addDependency(
                        release_name, previous_layer_return);

                    endpoint->workspace_allocator =
                        serial_graph_family_workspace_allocator_;
                    if (!endpoint->workspace_allocator)
                    {
                        throw std::runtime_error(
                            "Mapped ExpertOverlay follower could not create its serial device workspace owner");
                    }
                    WorkspaceSizingHints mapped_hints;
                    mapped_hints.max_seq_len = physical_rows;
                    mapped_hints.serial_family_max_rows = physical_rows;
                    mapped_hints.d_model = d_model;
                    mapped_hints.batch_size = 1;
                    mapped_hints.graph_family_policy =
                        WorkspaceGraphFamilyPolicy::
                            SerialDeviceFamilyExactParticipant;
                    if (!endpoint->workspace_allocator->allocateForGraph(
                            *endpoint->graph, mapped_hints))
                    {
                        throw std::runtime_error(
                            "Mapped ExpertOverlay follower could not allocate graph workspace");
                    }

                    IWorkerGPUContext *const worker =
                        participantWorkerContext(target_device);
                    GraphExecutorConfig executor_config;
                    executor_config.default_device = target_device;
                    executor_config.worker_gpu_context_resolver =
                        [worker, target_device](DeviceId requested)
                            -> IWorkerGPUContext *
                        {
                            return requested == target_device
                                       ? worker
                                       : nullptr;
                        };
                    executor_config.worker_gpu_context_uses_process_pool =
                        false;
                    endpoint->executor =
                        std::make_unique<DeviceGraphExecutor>(
                            executor_config);
                    endpoint->executor->setArena(endpoint->arena.get());
                    endpoint->cache.perf_context =
                        "moe_overlay_mapped_follower_p" +
                        std::to_string(endpoint->participant_id) +
                        "_family" +
                        std::to_string(graph_family_ordinal) +
                        "_rows" + std::to_string(physical_rows);
                    endpoint->cache.steady_replay_host_policy =
                        DeviceGraphExecutor::GraphSegmentCache::
                            SteadyReplayHostPolicy::RetainedFullGraph;
                    if (!endpoint->cache.bindBorrowedCaptureStream(
                            worker,
                            participantWorkerStream(target_device),
                            target_device,
                            /*context_from_process_pool=*/false))
                    {
                        throw std::runtime_error(
                            "Mapped ExpertOverlay follower could not bind its exact participant stream");
                    }
                    worker->submitAndWait(
                        [&]
                        {
                            if (original_native_slot_geometry_)
                            {
                                const auto manager = config_.model_context
                                    ? config_.model_context->concreteWeightManager() : nullptr;
                                const auto memory = manager ? manager->physicalMemoryAuthority() : nullptr;
                                if (!memory || !config_.model_context)
                                    throw std::logic_error("Original follower terminal lost its admitted source/PMA");
                                const auto domain = rank_local_publication_state_->originalSlotDomain();
                                auto population=config_.model_context->borrowPreparedFollowerCaptureTerminals(
                                    {endpoint->participant_id,graph_family_ordinal,physical_rows,target_device},
                                    endpoint->cache.capture_stream,*original_native_slot_geometry_,memory);
                                endpoint->cache.bindPreparedFollowerCaptureTerminals(domain,std::move(population),memory);
                            }
                            if (!endpoint->cache.ensureCaptureOutputEvent(worker))
                            {
                                throw std::runtime_error(
                                    "Mapped ExpertOverlay follower could not allocate its setup-owned terminal fence event");
                            }
                        });
                    if(!gpu_runtime->finite_boundary_endpoint) {
                        // This is the original declared family owner, not a
                        // synthetic executor or a lookup of live placement.
                        gpu_runtime->finite_boundary_endpoint=endpoint;
                        gpu_runtime->finite_boundary_context=execution_contexts_.at(target_device);
                    }
                    shape.endpoints.push_back(std::move(endpoint));
                }
                result->mapped_gpu_shapes.push_back(std::move(shape));
            }

            /* Native capture waits for physical-fabric epoch installation.
             * Graphs, buffers, prepared engines, and borrowed streams are all
             * durable now; only the final graph-owned transfer branch identity
             * is intentionally unresolved until orchestration finishes the
             * topology-accounted lane BOM. */
            PerfStatsCollector::addCounter(
                "moe_overlay_participant_graph",
                "declared_mapped_follower_families",
                1.0,
                "model_setup",
                participantDeviceList(local_participants_),
                {{"participants", participantIdList(local_participants_)},
                 {"row_shapes", std::to_string(row_shapes.size())},
                 {"gpu_timeline_executables",
                  std::to_string(
                      result->mapped_gpu_shapes.size() *
                      static_cast<size_t>(std::count_if(
                          bindings.begin(),
                          bindings.end(),
                          [](const auto &binding)
                          {
                              return binding.participant->device.is_gpu();
                          })))},
                 {"cpu_endpoints",
                  std::to_string(result->mapped_cpu_endpoints.size())},
                 {"graph_family", std::to_string(graph_family_ordinal)},
                 {"source_layers", std::to_string(spec.source_layers.size())},
                 {"native_capture", "deferred_until_physical_fabric"},
                 {"host_sparse_stages", "0"}});
            return result;
        }

        std::string previous_protocol_node;
        for (const int layer : spec.source_layers)
        {
            if (layer < 0 ||
                layer >= config_.model_context->totalBlockCount())
            {
                throw std::invalid_argument(
                    "Participant retained graph names an invalid source layer " +
                    std::to_string(layer));
            }
            for (size_t tier_index = 0;
                 tier_index < config_.placement_plan->routed_tiers.size();
                 ++tier_index)
            {
                const auto &tier =
                    config_.placement_plan->routed_tiers[tier_index];
                std::vector<int> targets =
                    owner_map_->participantIdsForTier(
                        static_cast<int>(tier_index));
                std::sort(targets.begin(), targets.end());

                std::vector<int> participants;
                for (const int target : targets)
                {
                    const auto target_mask =
                        owner_map_->expertMaskForParticipant(
                            layer, target, num_experts);
                    if (!hasActiveMask(target_mask))
                        continue;
                    const auto *descriptor =
                        owner_map_->participantForId(target);
                    if (!descriptor ||
                        !descriptor->world_rank_known)
                    {
                        throw std::runtime_error(
                            "Participant rank batch target has no authenticated MPI owner");
                    }
                    if (descriptor->domain_name ==
                            config_.placement_plan->continuation_domain &&
                        descriptor->world_rank ==
                            source_world_rank)
                    {
                        continue;
                    }
                    if (descriptor->world_rank != local_world_rank)
                        continue;
                    if (std::find(
                            local_participant_ids.begin(),
                            local_participant_ids.end(),
                            target) == local_participant_ids.end())
                    {
                        throw std::logic_error(
                            "Participant rank batch addresses an endpoint absent from the rank-local topology");
                    }
                    participants.push_back(target);
                }
                if (participants.empty())
                    continue;
                std::sort(participants.begin(), participants.end());

                const int routed_domain_ordinal =
                    domain_ordinal(tier.domain);
                auto transport = transport_for_group(
                    static_cast<int>(tier_index),
                    routed_domain_ordinal,
                    participants);

                std::vector<
                    std::shared_ptr<MoEOverlayCollectiveWorkspace>>
                    group_workspaces;
                std::vector<std::shared_ptr<MoEOverlaySparseRows>>
                    dispatch_inbound;
                std::vector<std::shared_ptr<MoEOverlayReturnRows>>
                    local_outputs;
                group_workspaces.reserve(participants.size());
                dispatch_inbound.reserve(participants.size());
                local_outputs.reserve(participants.size());
                for (const int participant : participants)
                {
                    auto workspace =
                        workspace_for_participant(participant);
                    group_workspaces.push_back(workspace);
                    dispatch_inbound.push_back(
                        std::make_shared<MoEOverlaySparseRows>(
                            transport->hasSharedRowStorage()
                                ? transport->sharedDispatchRows(participant)
                                : workspace->dispatchReceive(
                                      layer,
                                      static_cast<int>(tier_index))));
                    local_outputs.push_back(
                        std::make_shared<MoEOverlayReturnRows>(
                            transport->hasSharedRowStorage()
                                ? transport->sharedReturnRows(participant)
                                : workspace->localExpertOutput(
                                      layer,
                                      static_cast<int>(tier_index))));
                }

                const auto make_rank_batch_key =
                    [&](MoEOverlayCollectiveDirection direction)
                {
                    if (spec.role ==
                        ParticipantGraphFamilyRole::MTPDraft)
                    {
                        return makeMTPMoEOverlayRankBatchKey(
                            1,
                            0,
                            spec.mtp_graph_depth,
                            layer,
                            static_cast<int>(tier_index),
                            routed_domain_ordinal,
                            source_world_rank,
                            local_world_rank,
                            direction);
                    }
                    return makeMoEOverlayRankBatchKey(
                        1,
                        0,
                        ExpertHistogramSource::DecodeToken,
                        layer,
                        static_cast<int>(tier_index),
                        routed_domain_ordinal,
                        source_world_rank,
                        local_world_rank,
                        direction);
                };

                MoERankBatchDispatchStage::Params dispatch_params;
                dispatch_params.device_id = DeviceId::cpu();
                dispatch_params.endpoint_role =
                    MoERankBatchEndpointRole::RemoteTarget;
                dispatch_params.transport = transport;
                dispatch_params.key = make_rank_batch_key(
                    MoEOverlayCollectiveDirection::Dispatch);
                dispatch_params.source_participant =
                    continuation_root;
                dispatch_params.participant_ids = participants;
                dispatch_params.inbound_rows =
                    dispatch_inbound;
                dispatch_params.seq_len = row_capacity;
                dispatch_params.top_k = top_k;
                dispatch_params.d_model = d_model;
                dispatch_params.tier_index =
                    static_cast<int>(tier_index);

                const std::string stem =
                    "participant_layer" +
                    std::to_string(layer) +
                    "_tier" + std::to_string(tier_index) +
                    "_rank" + std::to_string(local_world_rank);
                const std::string dispatch_name =
                    stem + "_batch_dispatch";
                result->graph->addNode(
                    dispatch_name,
                    ComputeStageFactory::createMoERankBatchDispatch(
                        dispatch_params),
                    DeviceId::cpu());
                if (!previous_protocol_node.empty())
                {
                    result->graph->addDependency(
                        dispatch_name,
                        previous_protocol_node);
                }

                std::vector<std::string> local_names;
                local_names.reserve(participants.size());
                struct DeferredLocalCompletion
                {
                    MoELocalExpertStage *producer = nullptr;
                    DeviceId device = DeviceId::invalid();
                    int participant = -1;
                };
                std::vector<DeferredLocalCompletion>
                    deferred_local_completions;
                deferred_local_completions.reserve(participants.size());
                for (size_t participant_index = 0;
                     participant_index < participants.size();
                     ++participant_index)
                {
                    const int target =
                        participants[participant_index];
                    const auto *target_descriptor =
                        owner_map_->participantForId(target);
                    const auto local_participant_it =
                        std::find_if(
                            local_participants_.begin(),
                            local_participants_.end(),
                            [target](const auto *participant)
                            {
                                return participant &&
                                       participant->participant_id ==
                                           target;
                            });
                    if (!target_descriptor ||
                        local_participant_it ==
                            local_participants_.end() ||
                        (*local_participant_it)->device !=
                            target_descriptor->device)
                    {
                        throw std::runtime_error(
                            "Participant rank batch local endpoint topology changed during graph construction");
                    }

                    const DeviceId target_device =
                        target_descriptor->device;
                    MoELocalExpertStage::Params local_params;
                    local_params.device_id = target_device;
                    local_params.input_rows_lifetime =
                        dispatch_inbound[participant_index];
                    local_params.output_rows_lifetime =
                        local_outputs[participant_index];
                    local_params.workspace_lifetime =
                        group_workspaces[participant_index];
                    local_params.num_experts = num_experts;
                    local_params.top_k = top_k;
                    local_params.d_model = d_model;
                    local_params.expert_intermediate =
                        expert_intermediate;
                    local_params.layer_idx = layer;
                    local_params.expert_mask =
                        owner_map_->expertMaskForParticipant(
                            layer, target, num_experts);
                    local_params.prepared_store =
                        prepared_store_.get();
                    local_params.physical_memory_authority = prepared_store_
                        ? prepared_store_->physicalMemoryAuthority() : nullptr;
                    local_params.expert_registry = &registry;
                    local_params.runtime_participant_index =
                        target;
                    local_params.graph_row_capacity =
                        static_cast<size_t>(row_capacity);
                    local_params.completion_policy =
                        target_device.is_gpu()
                            ? MoELocalExpertStage::CompletionPolicy::
                                  DeferredExplicitStage
                            : MoELocalExpertStage::CompletionPolicy::Inline;
                    local_params.worker_gpu_context =
                        participantWorkerContext(target_device);
                    local_params.serial_compact_buffer_arena =
                        serialCompactBufferArenaForParticipant(
                            **local_participant_it);
                    local_params.expert_weight_resolution_policy =
                        MoELocalExpertStage::
                            ExpertWeightResolutionPolicy::
                                RegistryOnly;

                    (void)registry.populateExpertEnginesForParticipant(
                        tier.domain,
                        target_device,
                        target_descriptor->world_rank,
                        target_descriptor->domain_participant_index,
                        layer,
                        num_experts,
                        local_params.prepared_gate_gemm,
                        local_params.prepared_up_gemm,
                        local_params.prepared_down_gemm);
                    if (!MoELocalExpertStage::
                             prepareExpertGemmEngines(local_params))
                    {
                        throw std::runtime_error(
                            "Participant rank batch has incomplete prepared expert engines for layer " +
                            std::to_string(layer) + " participant " +
                            std::to_string(target));
                    }

                    auto endpoint =
                        config_.participant_residency->endpoint(
                            target);
                    if (!endpoint)
                    {
                        throw std::runtime_error(
                            "Participant rank batch is missing prepared-bank endpoint p" +
                            std::to_string(target));
                    }
                    std::vector<MoEOverlayPreparedExpertPayload>
                        prepared_payloads;
                    std::string residency_error;
                    if (!resolveMoEOverlayPreparedExpertPayloads(
                            registry,
                            *target_descriptor,
                            layer,
                            num_experts,
                            local_params.expert_mask,
                            prepared_payloads,
                            &residency_error) ||
                        !config_.participant_residency
                             ->registerInitialLayer(
                                 target,
                                 layer,
                                 local_params.expert_mask,
                                 prepared_payloads,
                                 &residency_error))
                    {
                        throw std::runtime_error(
                            "Participant rank batch could not install initial prepared residency bank for layer " +
                            std::to_string(layer) + " participant " +
                            std::to_string(target) + ": " +
                            residency_error);
                    }
                    local_params.overlay_participant_residency =
                        std::move(endpoint);

                    auto local_stage =
                        ComputeStageFactory::createMoELocalExpert(
                            local_params);
                    auto *const local_stage_ptr =
                        dynamic_cast<MoELocalExpertStage *>(
                            local_stage.get());
                    if (!local_stage_ptr)
                    {
                        throw std::runtime_error(
                            "Participant rank batch factory returned a non-local-expert stage");
                    }
                    if (target_device.is_gpu())
                    {
                        local_stage->setGPUStream(
                            participantWorkerStream(
                                target_device));
                        deferred_local_completions.push_back(
                            DeferredLocalCompletion{
                                .producer = local_stage_ptr,
                                .device = target_device,
                                .participant = target,
                            });
                    }
                    const std::string local_name =
                        stem + "_p" + std::to_string(target) +
                        "_local_expert";
                    result->graph->addNode(
                        local_name,
                        std::move(local_stage),
                        target_device);
                    result->graph->addDependency(
                        local_name,
                        dispatch_name);
                    local_names.push_back(local_name);
                }

                std::vector<std::string> completion_names;
                completion_names.reserve(
                    deferred_local_completions.size());
                for (const auto &deferred :
                     deferred_local_completions)
                {
                    MoELocalExpertCompletionStage::Params
                        completion_params;
                    completion_params.device_id = deferred.device;
                    completion_params.producer = deferred.producer;
                    auto completion_stage =
                        ComputeStageFactory::
                            createMoELocalExpertCompletion(
                                completion_params);
                    completion_stage->setGPUStream(
                        participantWorkerStream(
                            deferred.device));

                    const std::string completion_name =
                        stem + "_p" +
                        std::to_string(deferred.participant) +
                        "_local_expert_completion";
                    result->graph->addNode(
                        completion_name,
                        std::move(completion_stage),
                        deferred.device);

                    /*
                     * Every participant submission must be queued before the
                     * first host-visible output wait. This all-submit barrier
                     * is what turns distinct per-device streams into actual
                     * overlap rather than a series of submit/wait pairs.
                     */
                    for (const auto &local_name : local_names)
                    {
                        result->graph->addDependency(
                            completion_name,
                            local_name);
                    }
                    completion_names.push_back(completion_name);
                }

                std::vector<
                    std::shared_ptr<const MoEOverlayReturnRows>>
                    outbound_rows;
                outbound_rows.reserve(local_outputs.size());
                for (const auto &rows : local_outputs)
                    outbound_rows.push_back(rows);

                MoERankBatchReturnReduceStage::Params return_params;
                return_params.device_id = DeviceId::cpu();
                return_params.endpoint_role =
                    MoERankBatchEndpointRole::RemoteTarget;
                return_params.transport = transport;
                return_params.key = make_rank_batch_key(
                    MoEOverlayCollectiveDirection::ReturnReduce);
                return_params.continuation_participant =
                    continuation_root;
                return_params.participant_ids = participants;
                return_params.outbound_rows =
                    std::move(outbound_rows);
                return_params.seq_len = row_capacity;
                return_params.d_model = d_model;

                const std::string return_name =
                    stem + "_batch_return";
                result->graph->addNode(
                    return_name,
                    ComputeStageFactory::
                        createMoERankBatchReturnReduce(
                            return_params),
                    DeviceId::cpu());
                const auto &return_dependencies =
                    completion_names.empty()
                        ? local_names
                        : completion_names;
                for (const auto &dependency : return_dependencies)
                {
                    result->graph->addDependency(
                        return_name,
                        dependency);
                }
                previous_protocol_node = return_name;
            }
        }

        if (result->graph->size() == 0)
        {
            throw std::runtime_error(
                "Participant graph contains no sparse protocol stages");
        }
        result->workspace_allocator =
            serial_graph_family_workspace_allocator_;
        WorkspaceSizingHints hints;
        hints.max_seq_len = row_capacity;
        hints.serial_family_max_rows = row_capacity;
        hints.d_model = d_model;
        hints.batch_size = 1;
        hints.graph_family_policy =
            WorkspaceGraphFamilyPolicy::
                SerialDeviceFamilyExactParticipant;
        if (!result->workspace_allocator->allocateForGraph(
                *result->graph, hints))
        {
            throw std::runtime_error(
                "Failed to allocate participant graph workspace");
        }

        if (std::any_of(
                local_participants_.begin(),
                local_participants_.end(),
                [](const auto *participant)
                {
                    return participant && participant->device.is_gpu();
                }))
        {
            for (const auto &node_name :
                 result->graph->getExecutionOrder())
            {
                auto *node = result->graph->getNode(node_name);
                if (!node || !node->stage)
                    continue;
                auto *local_stage =
                    dynamic_cast<MoELocalExpertStage *>(node->stage.get());
                if (!local_stage)
                    continue;
                if (!local_stage->preparePersistentBuffers())
                {
                    throw std::runtime_error(
                        "Failed to prepare participant GPU sparse buffers");
                }
            }
        }

        std::string retained_plan_error;
        if (!executor_.prepareRetainedMultiDeviceExecutionPlan(
                *result->graph,
                execution_contexts_,
                result->execution_plan,
                &retained_plan_error))
        {
            throw std::runtime_error(
                "Failed to seal participant multi-device execution plan: " +
                retained_plan_error);
        }

        PerfStatsCollector::addCounter(
            "moe_overlay_participant_graph",
            "materialized_graphs",
            1.0,
            "model_setup",
            participantDeviceList(local_participants_),
            {
                {"participants", participantIdList(local_participants_)},
                {"row_capacity", std::to_string(row_capacity)},
                {"live_rows_source", "sparse_packet"},
                {"allocation_policy", "setup_only"},
                {"immutable", "true"},
                {"stages", std::to_string(result->graph->size())},
                {"dependency_waves",
                 std::to_string(result->execution_plan.waves.size())},
                {"parallel_gpu_waves",
                 std::to_string(
                     result->execution_plan.concurrent_gpu_wave_count)},
                {"max_wave_width",
                 std::to_string(result->execution_plan.max_wave_width)},
                {"graph_role",
                 spec.role == ParticipantGraphFamilyRole::Main
                     ? "main"
                     : "mtp_draft"},
                {"mtp_graph_depth",
                 std::to_string(spec.mtp_graph_depth)},
                {"source_layer_count",
                 std::to_string(spec.source_layers.size())},
            });
        return result;
    }

} // namespace llaminar2
