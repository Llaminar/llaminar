/**
 * @file Qwen35MoEGraph.cpp
 * @brief Qwen 3.5 MoE production graph builder implementation.
 *
 * The builder keeps participant-local compute declarative while lowering
 * explicit sparse dispatch and return collective boundaries for heterogeneous
 * expert-overlay deployments. Dense continuation remains local to each graph;
 * cross-rank expert work is addressed by logical participant identity so CUDA,
 * ROCm, and CPU/NUMA endpoints can share one production model topology.
 * Native-capture ownership follows every declared service boundary: portable
 * inter-host MPI and colocated CPU work share the typed ticket lifecycle.
 * Ordinary-prefill CPU input publication precedes independent shared GPU FFN
 * work. Measured mapped decode also keeps shared FFN in the original packet's
 * producer unit. Result ingress follows the final canonical-bank writer.
 * Independent rank-batch sends and colocated expert compute share a fork;
 * ordered return publication is the join, not a prerequisite for local compute.
 * CPU ExpertOverlay graphs adopt exact participant registry engines and shared
 * lifetimes; graph-family construction never repacks admitted expert weights.
 * CPU serial storage declarations survive admission and runtime projection.
 * Per-layer CPU packet roots retain their own exact backing allocation claims.
 * Enabled GPU setup diagnostics attribute complete transfer-profile scans as
 * nested FFN host work, without timing CPU inference graph construction.
 */

#include "Qwen35MoEGraphInternal.h"

namespace llaminar2
{

    const Qwen35MoEGraph::RoutedForwardRecipe &Qwen35MoEGraph::prepareRoutedForwardRecipe(DeviceId device)
    {
        if (device == DeviceId::invalid() || device != config_.default_device)
            throw std::invalid_argument("Qwen35 routed template requires its original valid participant");
        const auto before = routed_forward_template_.preparations();
        const auto &recipe = routed_forward_template_.prepare([&] {
            RoutedForwardRecipe prepared{.source_device = device,
                .effective_device = device, .runtime = runtimePlanForGraph(config_)};
            prepared.source = prepared.runtime ? prepared.runtime->sourcePlanPtr()
                                               : config_.moe.routed_expert_plan;
            if (prepared.runtime)
            {
                const auto &domain = prepared.runtime->continuationDomain();
                const auto continuation = prepared.runtime->continuationDevice();
                const bool contains = std::any_of(domain.participants.begin(), domain.participants.end(),
                    [&](const auto &participant) { return participant.local_device == device; });
                if (continuation.is_valid() && continuation != device && !contains)
                    prepared.effective_device = continuation;
            }
            if (prepared.source && prepared.source->usesExpertOverlayAuthority())
            {
                const auto &residency = config_.moe.expert_overlay_participant_residency;
                const auto preparation = residency ? residency->projectionPreparation() : nullptr;
                const auto original_joint_map = preparation ? preparation->jointOwnerMap() : nullptr;
                if (original_joint_map)
                {
                    // Source preparation already sealed the native shard cuts.
                    // The whole-owner constructor deliberately rejects them;
                    // borrow the exact original publication instead of creating
                    // an alias that silently assigns an expert to one socket.
                    if (&residency->initialOwnerMap() != original_joint_map.get())
                        throw std::logic_error("Qwen CPU joint graph lost its original prepared owner map");
                    prepared.owners = original_joint_map;
                    prepared.reused_original_owners = true;
                    ++original_owner_map_reuses_;
                }
                else
                {
                    // Ordinary whole-expert preparation retains its original
                    // canonical ordering and preferred-placement semantics.
                    ++canonical_owner_map_build_attempts_;
                    auto canonical = std::make_shared<MoEExpertOwnerMap>(
                        MoEExpertOwnerMap::build(*prepared.source));
                    if (residency && residency->initialOwnerMap().participants() == canonical->participants() &&
                        residency->initialOwnerMap().owners() == canonical->owners())
                    {
                        prepared.owners = std::shared_ptr<const MoEExpertOwnerMap>(
                            residency, &residency->initialOwnerMap());
                        prepared.reused_original_owners = true;
                        ++original_owner_map_reuses_;
                    }
                    else
                    {
                        prepared.owners = std::move(canonical);
                    }
                }
                ++continuation_resolution_attempts_;
                prepared.continuation = resolveRoutedContinuationTopology(
                    *prepared.source, prepared.owners.get(), prepared.effective_device,
                    dynamic_cast<ILocalTPContext *>(config_.tp_ctx), config_.tp_device_idx,
                    config_.moe.overlay_mpi_ctx != nullptr,
                    config_.moe.overlay_mpi_ctx ? config_.moe.overlay_mpi_ctx->rank() : -1);
            }
            return prepared;
        });
        if (recipe.source_device != device)
            throw std::invalid_argument("Qwen35 routed template requires its original participant");
        if (routed_forward_template_.preparations() != before && recipe.owners)
        {
            PerfStatsCollector::addCounter("forward_graph", "immutable_owner_map_preparations", 1.0,
                "map", device.to_string());
            PerfStatsCollector::addCounter("forward_graph", "immutable_continuation_preparations", 1.0,
                "topology", device.to_string());
            if (recipe.reused_original_owners)
                PerfStatsCollector::addCounter("forward_graph", "immutable_original_owner_map_reuses", 1.0,
                    "map", device.to_string());
        }
        return recipe;
    }

    void Qwen35MoEGraph::prepareForwardGraphTemplate(DeviceId device)
    {
        if (device == DeviceId::invalid() || device != config_.default_device)
            throw std::invalid_argument("Qwen35 forward template requires its original valid participant");
        if (config_.n_layers < 0 || (config_.n_layers > 0 && !hasLayerWeightSource()))
            throw std::invalid_argument("Qwen35 forward template requires its original layer weight source");
        (void)prepareRoutedForwardRecipe(device);
        QwenGraphBase::prepareForwardGraphTemplate(device);
    }

    const Qwen35MoEGraph::RoutedForwardRecipe &Qwen35MoEGraph::routedForwardRecipe(DeviceId device)
    {
        if (!routed_forward_template_.prepared())
            (void)prepareRoutedForwardRecipe(config_.default_device);
        const auto &recipe = routed_forward_template_.recipe();
        if (device != recipe.source_device && device != recipe.effective_device)
            throw std::invalid_argument("Qwen35 routed template cannot borrow a foreign participant");
        return recipe;
    }

    const Qwen35MoEGraph::RoutedForwardRecipe &Qwen35MoEGraph::routedForwardRecipe(DeviceId device) const
    {
        const auto &recipe = routed_forward_template_.recipe();
        if (device != recipe.source_device && device != recipe.effective_device)
            throw std::invalid_argument("Qwen35 routed template cannot borrow a foreign participant");
        return recipe;
    }

    PreparedGraphTemplateStats Qwen35MoEGraph::forwardGraphTemplateStats() const
    {
        auto result = QwenGraphBase::forwardGraphTemplateStats();
        result.canonical_owner_map_builds = canonical_owner_map_build_attempts_;
        result.continuation_resolutions = continuation_resolution_attempts_;
        result.original_owner_map_reuses = original_owner_map_reuses_;
        result.routed_layer_specialization_attempts = routed_layer_specialization_attempts_;
        return result;
    }

    // =========================================================================
    // Constructors
    // =========================================================================

    bool Qwen35MoEGraph::requiresCapturedSparseReturn(const MoERoutedExpertPlacementPlan& plan,
        const MoEExpertOwnerMap& owners, int layer, MoERebalanceRuntimeMode mode,
        std::optional<std::string_view> completed_return)
    {
        if (layer < plan.first_model_layer || plan.first_model_layer < 0 || plan.continuation_domain.empty())
            throw std::logic_error("Captured sparse publication has no original layer/domain");
        const auto quota_layer = static_cast<std::size_t>(layer - plan.first_model_layer);
        std::vector<std::size_t> counts(plan.routed_tiers.size(), 0);
        for (const auto& owner : owners.owners()) {
            if (owner.layer_idx != layer) continue;
            if (owner.tier_idx < 0 || static_cast<std::size_t>(owner.tier_idx) >= counts.size() ||
                owner.domain_name != plan.routed_tiers[static_cast<std::size_t>(owner.tier_idx)].domain)
                throw std::logic_error("Captured sparse publication changed original tier ownership");
            ++counts[static_cast<std::size_t>(owner.tier_idx)];
        }
        if (std::all_of(counts.begin(), counts.end(), [](auto count) { return count == 0; }))
            throw std::logic_error("Captured sparse publication has no complete routed ownership");
        bool external = false;
        for (std::size_t tier = 0; tier < counts.size(); ++tier) {
            if (plan.routed_tiers[tier].domain == plan.continuation_domain) continue;
            const auto& quotas = plan.routed_tiers[tier].resolved_live_experts_per_layer;
            if (counts[tier] != 0) external = true;
            else if (mode == MoERebalanceRuntimeMode::Dynamic &&
                (quota_layer >= quotas.size() || quotas[quota_layer] != 0))
                throw std::logic_error("Mutable empty external tier requires its original frozen zero quota");
        }
        if (external && completed_return && completed_return->empty())
            throw std::runtime_error("Qwen35 MoE captured distributed LocalTP root has no completed remote sparse return");
        return external;
    }

    Qwen35MoEGraph::Qwen35MoEGraph(
        std::shared_ptr<ModelContext> model_ctx,
        std::shared_ptr<IMPIContext> mpi_ctx,
        const GraphConfig &config)
        : Qwen35Graph(std::move(model_ctx), std::move(mpi_ctx), config)
    {
        bindProjectionArena();
    }

    Qwen35MoEGraph::Qwen35MoEGraph(
        const GraphConfig &config,
        std::shared_ptr<IMPIContext> mpi_ctx)
        : Qwen35Graph(config, std::move(mpi_ctx))
    {
        // GraphBuilderRegistry constructs declarations before source injection.
        // An unbound declaration is legal; resolving/executing it is not.
    }

    void Qwen35MoEGraph::bindProjectionArena()
    {
        if (config_.moe.routed_compute_policy == RoutedExpertComputePolicy::TensorSharded && config_.default_device.is_gpu())
        {
            if (!model_ctx_ || !config_.tp_ctx || !config_.tp_ctx->isLocal())
                throw std::invalid_argument("GPU F-shard arena requires its original source and native TP declaration");
            const auto profile = ModelMemoryProfile::fromLoader(model_ctx_->concreteLoader());
            const auto assignment = config_.tp_config ? std::optional{config_.tp_config->forRank(config_.tp_device_idx)}
                                                     : std::optional<DeviceShardingAssignment>{};
            const int last = config_.pp_layer_offset + config_.n_layers - 1;
            const int retained_last = retainsMTPGraphCapacity(config_.mtp) && last == profile.mainLayerCount() - 1
                ? profile.totalLayerCount() - 1 : last;
            intermediate_shard_arena_ = MoEIntermediateShardArenaGeometry::resolve(profile,
                config_.tp_ctx->degree(), config_.tp_device_idx, assignment, config_.pp_layer_offset, retained_last,
                config_.moe.routed_expert_plan ? config_.moe.routed_expert_plan->gpu_expert_intermediate_tiling
                                              : GPUExpertIntermediateTiling::Coarse128,
                config_.mtpUsesReplicatedDenseSidecarBinding()
                    ? MTPShiftedKVHeadLayout::FullModelPerParticipant
                    : MTPShiftedKVHeadLayout::PrimaryTensorParallelShard);
            return;
        }
        if (config_.moe.routed_compute_policy != RoutedExpertComputePolicy::GateUpOwnedDownColumns) return;
        if (!model_ctx_ || !config_.tp_ctx || !config_.tp_ctx->isLocal())
            throw std::invalid_argument("Gate/up-owned MoE requires source metadata and a native local TP domain");
        projection_arena_ = MoEProjectionArenaGeometry::resolve(
            ModelMemoryProfile::fromGGUF(model_ctx_->concreteLoader().getModel()), config_.tp_ctx->degree());
    }

    void Qwen35MoEGraph::setModelContext(std::shared_ptr<IModelContext> model_ctx)
    {
        if ((projection_arena_ || intermediate_shard_arena_) && model_ctx_ != model_ctx)
            throw std::logic_error("Cannot replace the source identity of a bound projection graph");
        Qwen35Graph::setModelContext(std::move(model_ctx));
        bindProjectionArena();
    }

    const MoEProjectionArenaGeometry *Qwen35MoEGraph::projectionArena() const
    {
        if (config_.moe.routed_compute_policy != RoutedExpertComputePolicy::GateUpOwnedDownColumns) return nullptr;
        if (!projection_arena_)
            throw std::logic_error("Projection graph metadata must be bound before schema or execution resolution");
        return &*projection_arena_;
    }

    const MoEIntermediateShardArenaGeometry *Qwen35MoEGraph::intermediateShardArena() const
    {
        if (config_.moe.routed_compute_policy != RoutedExpertComputePolicy::TensorSharded || !config_.default_device.is_gpu()) return nullptr;
        if (!intermediate_shard_arena_)
            throw std::logic_error("GPU F-shard source metadata must be bound before schema or execution resolution");
        return &*intermediate_shard_arena_;
    }

    void Qwen35MoEGraph::resetState(void *execution_stream)
    {
        Qwen35Graph::resetState(execution_stream);
        prefix_runtime_device_rehydration_pending_ = false;

        /* Routing evidence is model/residency-epoch state, not request state.
         * The maintenance authority alone rotates its RCU banks. Resetting it
         * here used to erase short-request evidence and could race an async
         * device drain while prefix or KV state was being cleared. */

        for (auto &[key, table] : moe_runtime_tables_)
        {
            (void)key;
            /*
             * Canonical main and MTP tables are model-lifetime residency state.
             * An LLEP child also carries the durable ticket, but its embedded
             * override banks are request-local and must return to their
             * immutable template before the next request follows the parent.
             */
            if (table &&
                (!table->usesOverlayEpochTicket() ||
                 table->overlayPlacementSource() != nullptr))
                table->restoreInitialRuntimeState(execution_stream);
        }

        /*
         * Transfer directories are model-lifetime allocation authorities.
         * Their entries may back a durable ticket-selected bank even after a
         * transient CurrentBatchLLEP table has been reset. The device allocator
         * derives liveness from the runtime family and uses compare-and-replace
         * generations, so stale request-local occupants are safely reusable;
         * clearing the shared directory here would instead invalidate durable
         * descriptors still selected by the next request's epoch ticket.
         */
    }

    void Qwen35MoEGraph::resetPrefixCacheRuntimeStateWithoutSnapshot(
        void *execution_stream)
    {
        /*
         * A cache block without a portable MoE payload owns no request-local
         * placement changes. Reset only transient CurrentBatchLLEP tables.
         * Epoch-ticketed placement remains owned by the live model-lifetime RCU
         * authority and is intentionally independent of prefix-cache state.
         */
        Qwen35Graph::resetState(execution_stream);
        prefix_runtime_device_rehydration_pending_ = false;

        /* Prefix reset does not own model-lifetime routing evidence. The
         * residency maintenance authority rotates that state independently. */

        /*
         * Graph-side bindings, auxiliary streams, and transfer-slot
         * directories own model-lifetime pointer identities captured by stage
         * nodes. Prefix reset must preserve those owners. Non-ticketed runtime
         * placement claims are request state; restoreInitialRuntimeState()
         * removes those transient claims while retaining stable pointer
         * identities. Ticketed canonical placement is not reset, while a
         * ticketed LLEP child is reset because its source pointer identifies
         * the durable parent independently of its embedded override banks.
         *
         * Clearing these maps used to free directories underneath existing
         * graph stages and then assume a later graph build would recreate
         * them.  ComputeGraph nodes outlive replay executables, so that
         * assumption left dangling transfer pointers and made prefix/MTP
         * failures dependent on whichever layer reused the freed slot first.
         */

        for (auto &[key, table] : moe_runtime_tables_)
        {
            (void)key;
            if (table &&
                (!table->usesOverlayEpochTicket() ||
                 table->overlayPlacementSource() != nullptr))
                table->restoreInitialRuntimeState(execution_stream);
        }
    }

    /**
     * @brief Locate the domain-wide decode maintenance binding for a device.
     *
     * The graph builder can create two independent classes of graph-side
     * transfer binding when Dynamic residency maintenance and current-batch
     * LLEP are both selected explicitly: one domain-wide decode-maintenance
     * binding and one or more layer-local prefill-transfer bindings. The async
     * maintenance graph is a decode-time control loop, so it must use the
     * decode binding even when a prefill binding was inserted later into the
     * unordered binding map.
     */
    const Qwen35MoEGraph::GraphSideRebalanceBinding *
    Qwen35MoEGraph::findDeviceMoERebalanceMaintenanceBinding(DeviceId device) const
    {
        const GraphSideRebalanceBinding *selected = nullptr;
        for (const auto &[key, binding] : moe_graph_rebalance_bindings_)
        {
            (void)key;
            if (binding.device_id != device)
            {
                continue;
            }
            if (binding.role != GraphSideRebalanceBindingRole::DecodeMaintenance)
            {
                continue;
            }
            if (binding.workspace_name.rfind("moe_device_rebalance_", 0) != 0)
            {
                throw std::runtime_error(
                    "Qwen35 MoE decode maintenance binding for " +
                    device.to_string() +
                    " points at a non-decode workspace: " +
                    binding.workspace_name);
            }
            if (selected)
            {
                throw std::runtime_error(
                    "Qwen35 MoE graph-side rebalance has multiple decode maintenance bindings for " +
                    device.to_string() +
                    "; refusing unordered maintenance graph selection");
            }
            selected = &binding;
        }

        return selected;
    }

    DeviceMoECurrentBatchLLEPEvidenceSource
    Qwen35MoEGraph::deviceMoECurrentBatchLLEPEvidenceSource(
        DeviceId device) const
    {
        DeviceMoECurrentBatchLLEPEvidenceSource selected{};
        const IMoERuntimeTable *selected_table = nullptr;

        for (const auto &[key, binding] : moe_graph_rebalance_bindings_)
        {
            (void)key;
            if (binding.device_id != device ||
                binding.role !=
                    GraphSideRebalanceBindingRole::CurrentBatchLLEPTransfer)
            {
                continue;
            }
            if (!binding.moe_runtime_table ||
                binding.moe_runtime_table->layerCount() <= 0)
            {
                throw std::logic_error(
                    "Qwen35 MoE current-batch LLEP evidence binding has no "
                    "complete runtime table for " +
                    device.to_string());
            }

            const auto *runtime_layers =
                binding.moe_runtime_table->deviceLayerState(
                    binding.moe_runtime_table->firstModelLayer());
            const int layer_count =
                binding.moe_runtime_table->layerCount();
            if (!runtime_layers)
            {
                throw std::logic_error(
                    "Qwen35 MoE current-batch LLEP evidence binding has a "
                    "null device runtime table for " +
                    device.to_string());
            }
            if (selected_table &&
                (selected_table != binding.moe_runtime_table ||
                 selected.runtime_layers_device != runtime_layers ||
                 selected.layer_count != layer_count ||
                 selected.expert_payload_slot_bytes !=
                     binding.collective_payload_slot_bytes))
            {
                throw std::logic_error(
                    "Qwen35 MoE current-batch LLEP bindings disagree on the "
                    "canonical runtime table or expert payload extent for " +
                    device.to_string());
            }
            if (binding.collective_payload_slot_bytes == 0)
            {
                throw std::logic_error(
                    "Qwen35 MoE current-batch LLEP evidence binding has a "
                    "zero expert payload extent for " +
                    device.to_string());
            }

            selected_table = binding.moe_runtime_table;
            selected = DeviceMoECurrentBatchLLEPEvidenceSource{
                .runtime_layers_device = runtime_layers,
                .layer_count = layer_count,
                .expert_payload_slot_bytes =
                    binding.collective_payload_slot_bytes,
            };
        }

        return selected;
    }

    bool Qwen35MoEGraph::finalizeMoEOverlayDeviceControllerRuntime(
        DeviceId device,
        void *publication_stream)
    {
        if (!device.is_gpu() || !publication_stream)
        {
            LOG_ERROR(
                "Qwen35 MoE initial controller runtime publication requires "
                "one exact GPU and non-null stream");
            return false;
        }

        const auto runtime_plan = routedForwardRecipe(config_.default_device).runtime;
        const auto &placement_plan = runtime_plan
                                         ? runtime_plan->sourcePlanPtr()
                                         : config_.moe.routed_expert_plan;
        if (!placement_plan ||
            !placement_plan->usesExpertOverlayAuthority())
        {
            return true;
        }

        const MoERuntimeTableIdentity identity{
            .role = MoERuntimeTableRole::MainDecodeDurablePlacement,
            .mtp_depth = -1,
        };
        const auto table_it = moe_runtime_tables_.find(
            moeRuntimeTableKey(device, identity));
        if (table_it == moe_runtime_tables_.end() || !table_it->second ||
            table_it->second->layerCount() <= 0)
        {
            LOG_ERROR(
                "Qwen35 MoE retained controller runtime finalization cannot "
                "resolve its canonical main table on "
                << device.toString());
            return false;
        }

        const auto &owner_map = *routedForwardRecipe(device).owners;
        const int local_world_rank = config_.moe.overlay_mpi_ctx
                                         ? config_.moe.overlay_mpi_ctx->rank()
                                         : -1;
        const MoEExpertOwnerParticipant *selected = nullptr;
        for (const auto &participant : owner_map.participants())
        {
            if (participant.device != device ||
                (participant.world_rank_known && local_world_rank >= 0 &&
                 participant.world_rank != local_world_rank))
            {
                continue;
            }
            if (selected)
            {
                LOG_ERROR(
                    "Qwen35 MoE retained controller runtime maps one device "
                    "to multiple local overlay participants: "
                    << device.toString());
                return false;
            }
            selected = &participant;
        }
        if (!selected || selected->participant_id < 0 ||
            selected->domain_participant_index < 0 ||
            !selected->world_rank_known)
        {
            LOG_ERROR(
                "Qwen35 MoE retained controller runtime cannot resolve its "
                "local overlay participant on "
                << device.toString());
            return false;
        }

        const auto weight_manager =
            model_ctx_ ? model_ctx_->concreteWeightManager() : nullptr;
        if (!weight_manager || config_.moe.num_experts <= 0 ||
            config_.moe.top_k <= 0 || config_.d_model <= 0 ||
            config_.moe.intermediate_size <= 0)
        {
            LOG_ERROR(
                "Qwen35 MoE retained controller runtime has incomplete model "
                "geometry or no prepared-engine registry");
            return false;
        }

        auto *const table = table_it->second.get();
        auto &registry = weight_manager->expertGemmRegistry();
        std::uint64_t finalized_layers = 0u;
        for (int layer = table->firstModelLayer(); layer < table->endModelLayer(); ++layer)
        {
            const bool publication_missing =
                table->decodeRuntimePublicationRequired(layer) ||
                table->hostLayerState(layer).active_epoch == 0u;
            const auto expert_mask = owner_map.expertMaskForParticipant(
                layer,
                selected->participant_id,
                config_.moe.num_experts);
            if (intermediate_shard_arena_)
            {
                const auto declaration = adoptMoEOverlayJointGraphDeclaration(
                    *config_.moe.expert_overlay_participant_residency, registry,
                    prepared_weight_store_->modelId(), selected->participant_id, layer,
                    config_.moe.num_experts, config_.d_model, config_.moe.intermediate_size);
                const auto initial = declaration->acquireInitialConstruction();
                if (!initial) throw std::logic_error("GPU F-shard runtime finalization lost original construction source");
                publishInitialMoEIntermediateShardRuntimeLayer(*table, owner_map,
                    initial->layerForModelLayer(layer), layer, publication_stream);
                if (publication_missing) ++finalized_layers;
                continue;
            }
            if (projection_arena_)
            {
                const auto *fixed = table->fixedDownProjectionBank(layer);
                if (!fixed) throw std::logic_error("Projection finalization lost its fixed down bank");
                std::vector<MoEOverlayPreparedExpertPayload> payloads;
                std::string error;
                if (!resolveMoEOverlayPreparedExpertPayloads(registry, *selected, layer,
                        config_.moe.num_experts, expert_mask, payloads, &error, fixed->ownership()))
                    throw std::runtime_error("Projection finalization: " + error);
                publishInitialMoEProjectionRuntimeLayer(*table, owner_map, layer, payloads, publication_stream);
                if (publication_missing) ++finalized_layers;
                continue;
            }
            std::vector<ITensorGemm *> gate_gemms;
            std::vector<ITensorGemm *> up_gemms;
            std::vector<ITensorGemm *> down_gemms;
            (void)registry.populateExpertEnginesForParticipant(
                selected->domain_name,
                device,
                selected->world_rank,
                selected->domain_participant_index,
                layer,
                config_.moe.num_experts,
                gate_gemms,
                up_gemms,
                down_gemms);

            bool local_payload_complete =
                gate_gemms.size() ==
                    static_cast<std::size_t>(config_.moe.num_experts) &&
                up_gemms.size() ==
                    static_cast<std::size_t>(config_.moe.num_experts) &&
                down_gemms.size() ==
                    static_cast<std::size_t>(config_.moe.num_experts);
            for (int expert = 0;
                 local_payload_complete && expert < config_.moe.num_experts;
                 ++expert)
            {
                if (!expert_mask[static_cast<std::size_t>(expert)])
                    continue;
                local_payload_complete =
                    gate_gemms[static_cast<std::size_t>(expert)] != nullptr &&
                    up_gemms[static_cast<std::size_t>(expert)] != nullptr &&
                    down_gemms[static_cast<std::size_t>(expert)] != nullptr;
            }
            if (!local_payload_complete)
            {
                LOG_ERROR(
                    "Qwen35 MoE retained controller runtime cannot resolve "
                    "every local prepared expert for layer "
                    << layer << " participant="
                    << selected->participant_id << " device="
                    << device.toString());
                return false;
            }

            const auto domain_local_owners =
                domainLocalOwnerParticipantsFromMap(
                    owner_map,
                    layer,
                    config_.moe.num_experts,
                    selected->domain_name);
            const auto overlay_route_participants =
                ownerParticipantsFromMap(
                    owner_map,
                    layer,
                    config_.moe.num_experts);
            if (!initializeMaskedLocalDecodeRuntimeTable(
                    table,
                    layer,
                    config_.moe.num_experts,
                    config_.moe.top_k,
                    config_.d_model,
                    config_.moe.intermediate_size,
                    expert_mask,
                    selected->domain_participant_index,
                    static_cast<int>(
                        owner_map.participantIdsForTier(
                            selected->tier_idx)
                            .size()),
                    domain_local_owners,
                    overlay_route_participants,
                    gate_gemms,
                    up_gemms,
                    down_gemms,
                    publication_stream,
                    /*allow_existing_dynamic_bank=*/true,
                    "complete retained ExpertOverlay controller runtime"))
            {
                return false;
            }
            if (publication_missing)
                ++finalized_layers;
        }

        try
        {
            /*
             * A host-side recipe can be complete even when the graph that
             * first referenced a dormant NextN layer has never launched.
             * Close setup with one explicit stream-ordered authority handoff
             * before the device controller scans or mutates the family.
             */
            table->sealAndPublishCompleteInitialRuntimeState(
                publication_stream);
        }
        catch (const std::exception &error)
        {
            LOG_ERROR(
                "Qwen35 MoE retained controller runtime could not seal and "
                "publish every physical model layer on "
                << device.toString() << ": " << error.what());
            return false;
        }

        PerfStatsCollector::addCounter(
            "moe_overlay_controller",
            "retained_runtime_layers_finalized",
            static_cast<double>(finalized_layers),
            "model_setup",
            device.toString(),
            {{"participant", std::to_string(selected->participant_id)},
             {"retained_layers", std::to_string(table->layerCount())},
             {"publication_stream", "controller_ordered"},
             {"blocking", "false"}});
        return true;
    }

    void Qwen35MoEGraph::retireBorrowedExecutionStreams(DeviceId device)
    {
        /* One graph builder may retain tables for main decode, prefill, every
         * MTP depth, and transient LLEP roles. They all borrow streams from the
         * same device-context lifetime, so retirement is a device-wide terminal
         * transition across the complete retained family rather than a guess at
         * whichever table most recently published a histogram. */
        for (auto &[key, table] : moe_runtime_tables_)
        {
            (void)key;
            if (table && table->deviceId() == device)
                table->retireRuntimeHistogramProducerStreams();
        }
    }

    MoEOverlayDeviceControllerRuntimeBinding
    Qwen35MoEGraph::deviceMoEOverlayControllerRuntimeBinding(
        DeviceId device) const
    {
        if (!device.is_gpu())
            return {};
        const auto runtime_plan = runtimePlanForGraph(config_);
        const auto &placement_plan = runtime_plan
                                         ? runtime_plan->sourcePlanPtr()
                                         : config_.moe.routed_expert_plan;
        if (!placement_plan ||
            !placement_plan->usesExpertOverlayAuthority())
        {
            return {};
        }

        const MoERuntimeTableIdentity identity{
            .role = MoERuntimeTableRole::MainDecodeDurablePlacement,
            .mtp_depth = -1,
        };
        const auto table_it = moe_runtime_tables_.find(
            moeRuntimeTableKey(device, identity));
        if (table_it == moe_runtime_tables_.end() || !table_it->second ||
            table_it->second->layerCount() <= 0)
        {
            return {};
        }

        const auto &owner_map = *routedForwardRecipe(device).owners;
        const int local_world_rank = config_.moe.overlay_mpi_ctx
                                         ? config_.moe.overlay_mpi_ctx->rank()
                                         : -1;
        const MoEExpertOwnerParticipant *selected = nullptr;
        for (const auto &participant : owner_map.participants())
        {
            if (participant.device != device ||
                (participant.world_rank_known && local_world_rank >= 0 &&
                 participant.world_rank != local_world_rank))
            {
                continue;
            }
            if (selected)
            {
                throw std::logic_error(
                    "Qwen35 MoE controller runtime device maps to multiple local overlay participants: " +
                    device.to_string());
            }
            selected = &participant;
        }
        if (!selected || selected->participant_id < 0 ||
            selected->domain_participant_index < 0)
        {
            throw std::logic_error(
                "Qwen35 MoE controller runtime cannot resolve its global overlay participant: " +
                device.to_string());
        }

        auto *const table = table_it->second.get();
        const auto &runtime = table->hostLayerState(table->firstModelLayer());
        auto *const epoch_arena = table->overlayEpochArena();
        if (!epoch_arena || epoch_arena->deviceId() != device)
        {
            throw std::logic_error(
                "Qwen35 MoE controller runtime lost its canonical device epoch arena for " +
                device.to_string());
        }
        MoEOverlayDeviceControllerRuntimeBinding binding{
            .device = device,
            .runtime_layers_device = table->deviceLayerState(table->firstModelLayer()),
            .runtime_table_host = table,
            .service_telemetry_device =
                table->deviceOverlayServiceTelemetry(),
            .service_samples_device =
                table->deviceOverlayServiceTelemetrySample(table->firstModelLayer()),
            .overlay_participant_id = selected->participant_id,
            .domain_participant_id = runtime.participant_id,
            .domain_participant_count = runtime.participant_count,
            .layer_count = static_cast<std::uint32_t>(table->layerCount()),
            .expert_count = runtime.expert_count,
            .top_k = runtime.top_k,
            .epoch_control = epoch_arena->deviceControlAddress(),
            .maintenance_epoch =
                epoch_arena->deviceMaintenanceEpochAddress(),
            .maintenance_status =
                epoch_arena->deviceMaintenanceStatusAddress(),
            .first_model_layer = table->firstModelLayer(),
        };
        if (!binding.publicationValid() ||
            binding.domain_participant_id !=
                static_cast<std::uint32_t>(
                    selected->domain_participant_index))
        {
            throw std::logic_error(
                "Qwen35 MoE controller runtime identity disagrees with the overlay owner map for " +
                device.to_string());
        }
        return binding;
    }

    ComputeGraph Qwen35MoEGraph::buildDeviceMoERebalanceMaintenanceGraph(
        DeviceId device)
    {
        const GraphSideRebalanceBinding *selected_binding =
            findDeviceMoERebalanceMaintenanceBinding(device);
        if (!selected_binding)
        {
            return {};
        }

        const GraphSideRebalanceBinding &binding = *selected_binding;
        const bool transfer_slot_mode_enabled =
            deviceMoERebalanceModeUsesTransferSlots(binding.transfer_mode);
        if (!binding.collective_tp_ctx ||
            !binding.moe_runtime_table ||
            (transfer_slot_mode_enabled &&
             (!binding.local_transfer_slots ||
              binding.local_transfer_slot_count == 0 ||
              !binding.transfer_state)))
        {
            throw std::runtime_error(
                "Qwen35 MoE device rebalance maintenance graph has incomplete binding for " +
                device.to_string());
        }

        /*
         * Maintenance is one graph-owned transaction. PlanCopyApply snapshots
         * source bytes before transport, uses the fixed-capacity collective
         * payload lane, and applies arrivals before the terminal event is
         * recorded. The device controller can publish a no-work command, so
         * the host never needs to inspect an edge mask or select a follow-up
         * graph.
         */
        MoEDeviceRebalanceStage::Params params;
        params.device_id = binding.device_id;
        params.tp_ctx = binding.collective_tp_ctx;
        params.moe_runtime_table = binding.moe_runtime_table;
        params.tp_device_idx = binding.tp_device_idx;
        params.config = binding.config;
        // Only this authored Dynamic maintenance family may publish policy
        // decisions. Current-batch/static assignment stages keep None.
        if (config_.moe.rebalance_config.mode == MoERebalanceRuntimeMode::Dynamic)
            params.config.flags |= static_cast<uint32_t>(DeviceMoERebalanceFlags::CompletedDynamicDecisions);
        params.local_transfer_slots = binding.local_transfer_slots;
        params.local_transfer_slot_count = binding.local_transfer_slot_count;
        params.collective_payload_slot_bytes =
            binding.collective_payload_slot_bytes;
        params.collective_payload_slot_capacity =
            binding.collective_payload_slot_capacity;
        params.stage_name = "moe_device_rebalance_maintenance";
        params.workspace_name = binding.workspace_name;
        params.phase = DeviceMoERebalanceStagePhase::PlanCopyApply;
        params.transfer_mode = binding.transfer_mode;
        params.transfer_state = binding.transfer_state;
        params.join_transfer_stream_after_copy = true;

        auto *canonical_runtime =
            dynamic_cast<DeviceMoERuntimeTable *>(binding.moe_runtime_table);
        if (!canonical_runtime)
        {
            throw std::runtime_error(
                "Qwen35 MoE maintenance binding is not a device runtime table for " +
                device.to_string());
        }
        if (canonical_runtime->usesOverlayEpochTicket())
        {
            const auto epoch_binding =
                deviceMoEOverlayEpochExecutionBinding(device);
            if (!epoch_binding ||
                canonical_runtime->overlayEpochArena() !=
                    epoch_binding.arena.get() ||
                canonical_runtime->overlayPlacementSource() != nullptr)
            {
                throw std::logic_error(
                    "Qwen35 MoE durable maintenance lost its canonical ExpertOverlay epoch authority for " +
                    device.to_string());
            }
            params.overlay_epoch_arena = epoch_binding.arena;
        }

        ComputeGraph graph;
        graph.addNode(
            params.stage_name,
            ComputeStageFactory::createMoEDeviceRebalance(params),
            binding.device_id);
        graph.setTerminalNode(params.stage_name);
        return graph;
    }

    void Qwen35MoEGraph::appendPrefixCacheFingerprintMaterial(PrefixFingerprintMaterial &material) const
    {
        material.moe.push_back({"graph.num_experts", std::to_string(config_.moe.num_experts)});
        material.moe.push_back({"graph.top_k", std::to_string(config_.moe.top_k)});
        // Wire compatibility is immutable metadata, independent of lazy table
        // creation. Old archive schemas must miss before payload restoration.
        material.moe.push_back({"graph.portable_runtime_version",
                                std::to_string(kMoEPrefixRuntimeVersion)});
        material.moe.push_back({"graph.owner_order",
                                routedExpertOwnerOrderToString(config_.moe.owner_order)});
        material.moe.push_back({"graph.owner_participant_index",
                                std::to_string(config_.moe.owner_participant_index)});
        material.moe.push_back({"graph.owner_participant_count",
                                std::to_string(config_.moe.owner_participant_count)});

        if (config_.moe.routed_expert_plan)
        {
            appendExpertOverlayPlanFingerprintFields(
                material.moe,
                *config_.moe.routed_expert_plan,
                "expert_overlay.plan");
        }
        else
        {
            material.moe.push_back({"expert_overlay.plan.enabled", "false"});
        }

        if (config_.moe.expert_overlay_runtime_plan)
        {
            appendExpertOverlayRuntimeFingerprintFields(
                material.moe,
                *config_.moe.expert_overlay_runtime_plan,
                "expert_overlay.runtime");
        }
        else
        {
            material.moe.push_back({"expert_overlay.runtime.enabled", "false"});
        }
    }

    size_t Qwen35MoEGraph::prefixCacheRuntimeStateCapacity() const
    {
        size_t table_bytes = 0u;
        for (const auto &[key, table] : moe_runtime_tables_)
        {
            if (!table || table->usesOverlayEpochTicket())
                continue;
            const int layers = table->layerCount();
            const int experts = table->expertCount();
            if (layers < 0 || experts < 0)
                throw std::logic_error("invalid MoE prefix serializer geometry");
            // Match the portable serializer below: length-prefixed table name,
            // two geometry words, seven words per layer, then eight words and
            // two 64-bit histogram values per expert. Mutable eligibility may
            // omit a table, but cannot make its serialized geometry larger.
            constexpr size_t expert_bytes = 8u * sizeof(uint32_t) + 2u * sizeof(uint64_t);
            constexpr size_t layer_header = 7u * sizeof(uint32_t);
            constexpr size_t table_header = 3u * sizeof(uint32_t);
            constexpr size_t archive_header = sizeof(kMoEPrefixRuntimeMagic) + 3u * sizeof(uint32_t);
            const auto limit = std::numeric_limits<size_t>::max();
            if (static_cast<size_t>(experts) > (limit - layer_header) / expert_bytes)
                throw std::overflow_error("MoE prefix expert geometry overflow");
            const size_t layer_bytes = layer_header + static_cast<size_t>(experts) * expert_bytes;
            if (key.size() > limit - table_header ||
                static_cast<size_t>(layers) > (limit - table_header - key.size()) / layer_bytes)
                throw std::overflow_error("MoE prefix layer geometry overflow");
            const size_t bytes = table_header + key.size() + static_cast<size_t>(layers) * layer_bytes;
            if (table_bytes > limit - archive_header || bytes > limit - archive_header - table_bytes)
                throw std::overflow_error("MoE prefix serializer capacity overflow");
            table_bytes += bytes;
        }
        return table_bytes == 0u ? 0u :
            sizeof(kMoEPrefixRuntimeMagic) + 3u * sizeof(uint32_t) + table_bytes;
    }

    bool Qwen35MoEGraph::capturePrefixCacheRuntimeState(std::vector<uint8_t> &state, void *stream)
    {
        /*
         * DeviceMoELayerRuntime itself is not a prefix-cache payload: it owns
         * process-local packed-weight descriptors and transfer-slot pointers.
         * Prefix blocks instead carry portable logical MoE runtime state:
         * owners, residency masks, local-compute masks, replica roles, epochs,
         * and decode histograms.  Restore resolves those logical entries back
         * to descriptors in the live runner and fails hard if required local
         * payloads are not resident.
         */
        state.clear();
        if (moe_runtime_tables_.empty())
            return true;

        struct CapturedTable
        {
            std::string key;
            uint32_t layers = 0;
            uint32_t experts = 0;
            std::vector<DeviceMoEPortableLayerRuntimeState> runtime_layers;
        };

        std::vector<CapturedTable> captured_tables;
        const auto expected_participant_count =
            expectedPrefixRuntimeParticipantCount(config_);
        for (auto &[key, table] : moe_runtime_tables_)
        {
            if (!table)
                continue;
            /*
             * Prefix blocks restore request state; durable ExpertOverlay
             * placement is live model state and may have advanced many epochs
             * since the block was captured. MTP sidecars share that exact main
             * placement authority, so excluding every ticketed table also
             * prevents stale embedded sidecar banks from entering the payload.
             */
            if (table->usesOverlayEpochTicket())
            {
                PerfStatsCollector::addCounter(
                    "prefix_cache",
                    "moe_durable_runtime_state_excluded_from_prefix",
                    1.0,
                    "prefix_cache",
                    config_.default_device.toString(),
                    {{"table", key}});
                continue;
            }
            std::vector<DeviceMoEPortableLayerRuntimeState> runtime_layers;
            if (!table->capturePortableRuntimeState(runtime_layers, stream))
            {
                LOG_ERROR("[Qwen35MoEGraph] Failed to capture prefix-cache MoE runtime state for " << key);
                return false;
            }
            if (runtime_layers.empty())
                continue;
            if (!portableRuntimeStateMatchesPrefixRestoreDomain(
                    key,
                    runtime_layers,
                    expected_participant_count,
                    config_.default_device))
            {
                continue;
            }
            CapturedTable captured;
            captured.key = key;
            captured.layers = static_cast<uint32_t>(runtime_layers.size());
            captured.experts = static_cast<uint32_t>(config_.moe.num_experts);
            captured.runtime_layers = std::move(runtime_layers);
            captured_tables.push_back(std::move(captured));
        }

        if (captured_tables.empty())
            return true;

        appendBytes(state, kMoEPrefixRuntimeMagic, sizeof(kMoEPrefixRuntimeMagic));
        appendU32(state, kMoEPrefixRuntimeVersion);
        appendU32(state, static_cast<uint32_t>(config_.moe.top_k));
        appendU32(state, static_cast<uint32_t>(captured_tables.size()));
        for (const auto &captured : captured_tables)
        {
            appendString(state, captured.key);
            appendU32(state, captured.layers);
            appendU32(state, captured.experts);
            for (const auto &layer : captured.runtime_layers)
            {
                appendI32(state, layer.model_layer);
                appendU32(state, layer.active_epoch);
                appendU32(state, layer.expert_count);
                appendU32(state, layer.top_k);
                appendU32(state, layer.participant_id);
                appendU32(state, layer.participant_count);
                appendU32(
                    state,
                    layer.requires_device_payload_rehydration);
                for (const auto &expert : layer.experts)
                {
                    appendI32(state, expert.logical_expert_id);
                    appendI32(state, expert.owner_participant);
                    appendI32(state, expert.local_slot);
                    appendU32(state, expert.flags);
                    appendU32(state, static_cast<uint32_t>(expert.local_compute));
                    appendU32(state, static_cast<uint32_t>(expert.replica_role));
                    appendU32(state, expert.resident_participant_mask);
                    appendI32(state, expert.overlay_route_participant);
                }
                for (uint64_t count : layer.selected_histogram)
                    appendU64(state, count);
                for (uint64_t count : layer.local_histogram)
                    appendU64(state, count);
            }
        }

        PerfStatsCollector::addCounter(
            "prefix_cache",
            "moe_portable_runtime_state_captures",
            1.0,
            "prefix_cache",
            config_.default_device.toString(),
            {{"tables", std::to_string(captured_tables.size())},
             {"bytes", std::to_string(state.size())}});
        return true;
    }

    PrefixCacheRuntimeRestoreResult
    Qwen35MoEGraph::restorePrefixCacheRuntimeState(
        std::span<const uint8_t> state,
        void *stream)
    {
        prefix_runtime_device_rehydration_pending_ = false;
        if (state.empty())
        {
            return PrefixCacheRuntimeRestoreResult{
                .restored = true,
            };
        }
        if (state.size() < sizeof(kMoEPrefixRuntimeMagic) ||
            std::memcmp(state.data(), kMoEPrefixRuntimeMagic, sizeof(kMoEPrefixRuntimeMagic)) != 0)
        {
            LOG_ERROR("[Qwen35MoEGraph] Refusing obsolete prefix-cache MoE runtime state: "
                      "MoE placement snapshots must not serialize runtime-local device descriptors");
            return {};
        }

        size_t offset = sizeof(kMoEPrefixRuntimeMagic);
        uint32_t version = 0;
        uint32_t top_k = 0;
        uint32_t table_count = 0;
        if (!readU32(state, offset, version) ||
            !readU32(state, offset, top_k) ||
            !readU32(state, offset, table_count))
        {
            LOG_ERROR("[Qwen35MoEGraph] Malformed prefix-cache MoE runtime state header");
            return {};
        }
        if (version != kMoEPrefixRuntimeVersion)
        {
            LOG_ERROR("[Qwen35MoEGraph] Unsupported prefix-cache MoE runtime state version "
                      << version);
            return {};
        }
        if (top_k != static_cast<uint32_t>(config_.moe.top_k))
        {
            LOG_ERROR("[Qwen35MoEGraph] Prefix-cache MoE histogram top-k mismatch: blob="
                      << top_k << " graph=" << config_.moe.top_k);
            return {};
        }

        // Parse and authenticate all stage identities before restoring any
        // table. A foreign later table or a truncated suffix must not overwrite
        // a valid earlier table merely because it appears first in the archive.
        struct ParsedTable
        {
            std::string key;
            MoERuntimeTable *table = nullptr;
            std::vector<DeviceMoEPortableLayerRuntimeState> layers;
        };
        if (table_count > moe_runtime_tables_.size())
        {
            LOG_ERROR("[Qwen35MoEGraph] Prefix-cache MoE archive exceeds the live table inventory");
            return {};
        }
        std::vector<ParsedTable> parsed_tables;
        parsed_tables.reserve(table_count);
        std::unordered_set<std::string> seen_keys;
        for (uint32_t table_idx = 0; table_idx < table_count; ++table_idx)
        {
            std::string key;
            uint32_t layers = 0;
            uint32_t experts = 0;
            if (!readString(state, offset, key) ||
                !readU32(state, offset, layers) ||
                !readU32(state, offset, experts))
            {
                LOG_ERROR("[Qwen35MoEGraph] Malformed prefix-cache MoE portable runtime table header");
                return {};
            }
            if (experts != static_cast<uint32_t>(config_.moe.num_experts))
            {
                LOG_ERROR("[Qwen35MoEGraph] Prefix-cache MoE portable runtime expert-count mismatch for "
                          << key << ": blob=" << experts
                          << " graph=" << config_.moe.num_experts);
                return {};
            }
            if (!seen_keys.insert(key).second)
            {
                LOG_ERROR("[Qwen35MoEGraph] Prefix-cache MoE archive repeats runtime table " << key);
                return {};
            }
            auto it = moe_runtime_tables_.find(key);
            if (it == moe_runtime_tables_.end() || !it->second)
            {
                LOG_ERROR("[Qwen35MoEGraph] Prefix-cache MoE portable runtime restore could not find runtime table "
                          << key);
                return {};
            }
            if (it->second->usesOverlayEpochTicket())
            {
                LOG_ERROR("[Qwen35MoEGraph] Prefix-cache MoE runtime state attempted to replace model-lifetime ExpertOverlay placement for "
                          << key);
                return {};
            }
            if (layers != static_cast<uint32_t>(it->second->layerCount()))
            {
                LOG_ERROR("[Qwen35MoEGraph] Prefix-cache MoE portable runtime layer-count mismatch for "
                          << key << ": blob=" << layers
                          << " table=" << it->second->layerCount());
                return {};
            }

            std::vector<DeviceMoEPortableLayerRuntimeState> runtime_layers;
            runtime_layers.resize(static_cast<size_t>(layers));
            for (uint32_t layer_idx = 0; layer_idx < layers; ++layer_idx)
            {
                auto &layer = runtime_layers[static_cast<size_t>(layer_idx)];
                if (!readI32(state, offset, layer.model_layer) ||
                    !readU32(state, offset, layer.active_epoch) ||
                    !readU32(state, offset, layer.expert_count) ||
                    !readU32(state, offset, layer.top_k) ||
                    !readU32(state, offset, layer.participant_id) ||
                    !readU32(state, offset, layer.participant_count) ||
                    !readU32(
                        state,
                        offset,
                        layer.requires_device_payload_rehydration))
                {
                    LOG_ERROR("[Qwen35MoEGraph] Malformed prefix-cache MoE portable runtime layer header");
                    return {};
                }
                if (layer.model_layer != it->second->firstModelLayer() + static_cast<int>(layer_idx) ||
                    layer.expert_count != experts ||
                    layer.top_k != static_cast<uint32_t>(config_.moe.top_k))
                {
                    LOG_ERROR("[Qwen35MoEGraph] Prefix-cache MoE portable runtime layer metadata mismatch for "
                              << key << " layer=" << layer_idx);
                    return {};
                }
                layer.experts.resize(static_cast<size_t>(experts));
                for (uint32_t expert_idx = 0; expert_idx < experts; ++expert_idx)
                {
                    auto &expert = layer.experts[static_cast<size_t>(expert_idx)];
                    int32_t logical_expert_id = -1;
                    int32_t owner_participant = -1;
                    int32_t local_slot = -1;
                    uint32_t local_compute = 0;
                    uint32_t replica_role = 0;
                    if (!readI32(state, offset, logical_expert_id) ||
                        !readI32(state, offset, owner_participant) ||
                        !readI32(state, offset, local_slot) ||
                        !readU32(state, offset, expert.flags) ||
                        !readU32(state, offset, local_compute) ||
                        !readU32(state, offset, replica_role) ||
                        !readU32(state, offset, expert.resident_participant_mask) ||
                        !readI32(state, offset, expert.overlay_route_participant))
                    {
                        LOG_ERROR("[Qwen35MoEGraph] Malformed prefix-cache MoE portable runtime expert payload");
                        return {};
                    }
                    expert.logical_expert_id = logical_expert_id;
                    expert.owner_participant = owner_participant;
                    expert.local_slot = local_slot;
                    expert.local_compute = local_compute != 0u ? 1u : 0u;
                    expert.replica_role = static_cast<uint8_t>(replica_role);
                }
                layer.selected_histogram.assign(static_cast<size_t>(experts), 0ULL);
                layer.local_histogram.assign(static_cast<size_t>(experts), 0ULL);
                layer.prefill_selected_histogram.assign(
                    static_cast<size_t>(experts), 0ULL);
                layer.prefill_local_histogram.assign(
                    static_cast<size_t>(experts), 0ULL);
                layer.grouped_verifier_selected_histogram.assign(
                    static_cast<size_t>(experts), 0ULL);
                layer.grouped_verifier_local_histogram.assign(
                    static_cast<size_t>(experts), 0ULL);
                for (uint32_t expert_idx = 0; expert_idx < experts; ++expert_idx)
                {
                    if (!readU64(state, offset, layer.selected_histogram[static_cast<size_t>(expert_idx)]))
                    {
                        LOG_ERROR("[Qwen35MoEGraph] Malformed prefix-cache MoE portable selected histogram payload");
                        return {};
                    }
                }
                for (uint32_t expert_idx = 0; expert_idx < experts; ++expert_idx)
                {
                    if (!readU64(state, offset, layer.local_histogram[static_cast<size_t>(expert_idx)]))
                    {
                        LOG_ERROR("[Qwen35MoEGraph] Malformed prefix-cache MoE portable local histogram payload");
                        return {};
                    }
                }
            }

            parsed_tables.push_back(ParsedTable{
                .key = std::move(key),
                .table = it->second.get(),
                .layers = std::move(runtime_layers),
            });
        }
        if (offset != state.size())
        {
            LOG_ERROR("[Qwen35MoEGraph] Prefix-cache MoE runtime state has trailing bytes");
            return {};
        }

        /*
         * Portable version 6 contains pointer-free, stage-bound request state.
         * Model-lifetime ExpertOverlay placement is selected by the shared RCU
         * ticket and rejected above, so an old prefix cannot rewind residency.
         * Transient payload descriptors are rehydrated from immutable owner
         * weights by a captured transaction, never from rolling slot contents.
         */
        uint32_t restored_tables = 0;
        bool requires_device_rehydration = false;
        bool placement_changed = false;
        for (const auto &parsed : parsed_tables)
        {
            const DeviceMoEPortableRuntimeRestoreResult table_restore =
                parsed.table->restorePortableRuntimeState(parsed.layers, stream);
            if (!table_restore)
            {
                LOG_ERROR("[Qwen35MoEGraph] Prefix-cache MoE portable runtime restore failed for " << parsed.key);
                return {};
            }
            placement_changed =
                placement_changed ||
                table_restore.placement_effect ==
                    DeviceMoEPortablePlacementEffect::Changed;
            requires_device_rehydration =
                requires_device_rehydration ||
                table_restore.requires_device_payload_rehydration;
            ++restored_tables;
        }

        prefix_runtime_device_rehydration_pending_ =
            requires_device_rehydration;
        LOG_INFO("[Qwen35MoEGraph] Prefix-runtime restore transaction"
                 << " device=" << config_.default_device.toString()
                 << " tables=" << restored_tables
                 << " placement_effect="
                 << (placement_changed ? "changed" : "unchanged")
                 << " device_payload_rehydration="
                 << (prefix_runtime_device_rehydration_pending_
                         ? "pending"
                         : "not_required"));
        PerfStatsCollector::addCounter(
            "prefix_cache",
            "moe_portable_runtime_state_restores",
            1.0,
            "prefix_cache",
            config_.default_device.toString(),
            {{"tables", std::to_string(restored_tables)},
             {"bytes", std::to_string(state.size())},
             {"placement_effect",
              placement_changed ? "changed" : "unchanged"},
             {"device_payload_rehydration",
              requires_device_rehydration ? "required" : "not_required"}});
        return PrefixCacheRuntimeRestoreResult{
            .restored = true,
            .placement_effect =
                placement_changed
                    ? PrefixCacheRuntimePlacementEffect::Changed
                    : PrefixCacheRuntimePlacementEffect::Unchanged,
            .device_rehydration_required = requires_device_rehydration,
        };
    }

    void Qwen35MoEGraph::completePrefixCacheRuntimeStateDeviceRehydration()
    {
        if (!prefix_runtime_device_rehydration_pending_)
        {
            throw std::runtime_error(
                "Qwen35 MoE prefix-runtime device rehydration completion was published without a pending restore transaction");
        }
        prefix_runtime_device_rehydration_pending_ = false;
        PerfStatsCollector::addCounter(
            "prefix_cache",
            "moe_portable_runtime_device_rehydrations",
            1.0,
            "prefix_cache",
            config_.default_device.toString());
    }

    std::string Qwen35MoEGraph::moeRuntimeTableKey(
        DeviceId device,
        const MoERuntimeTableIdentity &identity) const
    {
        const std::string stage_key = device.to_string() + "#layers_" +
            std::to_string(config_.pp_layer_offset) + "_" + std::to_string(config_.n_layers);
        switch (identity.role)
        {
        case MoERuntimeTableRole::MainDecodeDurablePlacement:
            if (identity.mtp_depth >= 0)
            {
                throw std::invalid_argument(
                    "Durable main-decode MoE runtime identity cannot carry an MTP depth");
            }
            return stage_key;
        case MoERuntimeTableRole::CurrentBatchLLEPPrefill:
            if (identity.mtp_depth >= 0)
            {
                throw std::invalid_argument(
                    "Current-batch LLEP prefill runtime identity cannot carry an MTP depth");
            }
            return stage_key + "#current_batch_llep_prefill";
        case MoERuntimeTableRole::MTPDepth:
            if (identity.mtp_depth < 0)
            {
                throw std::invalid_argument(
                    "MTP MoE runtime identity requires a non-negative depth");
            }
            return stage_key + "#mtp_depth" +
                   std::to_string(identity.mtp_depth);
        }
        throw std::logic_error("Unknown MoE runtime table role");
    }

    std::shared_ptr<MoELocalExpertSerialBufferArena>
    Qwen35MoEGraph::localExpertSerialBufferArenaForParticipant(
        DeviceId device,
        int participant,
        size_t required_row_capacity,
        std::optional<DeviceId> cpu_canonical_route_gpu_consumer)
    {
        if (!device.is_valid() || participant < 0 || required_row_capacity == 0 ||
            config_.d_model <= 0 || config_.moe.top_k <= 0 ||
            (cpu_canonical_route_gpu_consumer &&
             (!device.is_cpu() ||
              !cpu_canonical_route_gpu_consumer->is_gpu())))
        {
            throw std::invalid_argument(
                "Qwen35 MoE serial compact-buffer arena requires a valid device, "
                "participant, row capacity, d_model, and top_k");
        }

        /*
         * The orchestrator declares decode, prefill buckets, grouped verifier,
         * and MTP sidecar graphs as one serial family. Plan the largest row
         * geometry once, then allocate a separately coherent verifier-sized
         * family beside it. This retains setup-only stable addresses without
         * paying maximum-prefill transfer volume for decode.
         */
        const int prefill_rows = overlayPrefillSegmentRowCapacity(
            config_, device);
        const int verifier_rows =
            retainsMTPGraphCapacity(config_.mtp)
                ? std::max(
                      1,
                      resolveMTPRetainedTargetQueryRows(config_.mtp))
                : 1;
        const size_t planned_rows = static_cast<size_t>(
            std::max(prefill_rows, verifier_rows));
        const size_t planned_row_capacity = std::max(
            required_row_capacity,
            planned_rows);

        const std::string key =
            device.to_string() + "#overlay_participant" +
            std::to_string(participant);
        auto existing = moe_serial_local_expert_buffer_arenas_.find(key);
        if (existing != moe_serial_local_expert_buffer_arenas_.end())
        {
            if (!existing->second ||
                !existing->second->supports(
                    device,
                    required_row_capacity,
                    config_.d_model,
                    config_.moe.top_k))
            {
                throw std::logic_error(
                    "Qwen35 MoE graph requested compact local-expert storage beyond "
                    "the immutable serial-family plan for " +
                    key + ": requested_rows=" +
                    std::to_string(required_row_capacity) +
                    " planned_rows=" +
                    std::to_string(
                        existing->second
                            ? existing->second->rowCapacity()
                            : 0));
            }
            if (cpu_canonical_route_gpu_consumer)
            {
                /* This is also the immutable endpoint-identity check. The
                 * accessor never binds or registers storage after creation. */
                (void)existing->second->mappedCPUCanonicalRoutes(
                    *cpu_canonical_route_gpu_consumer);
            }
            return existing->second;
        }

        MoELocalExpertSerialBufferArena::Config arena_config;
        arena_config.device_id = device;
        arena_config.row_capacity = planned_row_capacity;
        arena_config.row_capacity_buckets =
            MoELocalExpertSerialBufferArena::powerOfTwoRowBucketsThrough(
                planned_row_capacity);
        arena_config.d_model = config_.d_model;
        arena_config.routing_top_k = config_.moe.top_k;
        if (device.is_cpu())
        {
            const auto memory=prepared_weight_store_?prepared_weight_store_->physicalMemoryAuthority():nullptr;
            if(!memory) throw std::logic_error("CPU serial graph roots require the existing prepared-store PMA");
            const auto &admission=requireCPUSerialExpertStorageAdmission(config_.cpu_serial_expert_storage,
                {memory->worldRank(),device},participant,config_.default_device);
            if(cpu_canonical_route_gpu_consumer && *cpu_canonical_route_gpu_consumer!=config_.default_device)
                throw std::invalid_argument("CPU canonical publication consumer differs from its admitted controller");
            arena_config.cpu_storage_layout=admission.envelope->projectLocal(prefill_rows,verifier_rows,
                cpu_canonical_route_gpu_consumer);
            if(required_row_capacity>arena_config.cpu_storage_layout->maximumRows())
                throw std::invalid_argument("CPU local expert requested rows outside its admitted storage declaration");
            arena_config.row_capacity=arena_config.cpu_storage_layout->maximumRows();
            arena_config.row_capacity_buckets.clear();
            for(const auto &family:arena_config.cpu_storage_layout->families())
                arena_config.row_capacity_buckets.push_back(family.rows);
            arena_config.physical_memory_authority=memory;
            // The admitted layout gives only the original reduction root
            // the combined canonical rows. Every member owns its packed rows.
            arena_config.cpu_canonical_route_storage = arena_config.cpu_storage_layout->canonical()
                ? MoELocalExpertSerialBufferArena::CPUCanonicalRouteStoragePolicy::RetainSerialMaximum
                : MoELocalExpertSerialBufferArena::CPUCanonicalRouteStoragePolicy::Disabled;
            arena_config.cpu_canonical_route_gpu_consumer =
                cpu_canonical_route_gpu_consumer;
            arena_config.cpu_grouped_scratch_storage =
                MoELocalExpertSerialBufferArena::
                    CPUGroupedScratchStoragePolicy::RetainSerialMaximum;
            arena_config.num_experts = config_.moe.num_experts;
            // The original admitted CPU cohort owns this member's local F.
            // Dense source geometry remains full-width elsewhere; using it for
            // these serial roots would contradict the exact frozen declaration.
            arena_config.expert_intermediate = arena_config.cpu_storage_layout->jointRoutes()
                ? arena_config.cpu_storage_layout->jointRoutes()->intermediate()
                : config_.moe.intermediate_size;
        }
        arena_config.logical_participant_id = participant;
        arena_config.debug_name =
            "moe_overlay_serial_compact." + key;
        auto arena = std::make_shared<MoELocalExpertSerialBufferArena>(
            std::move(arena_config));
        moe_serial_local_expert_buffer_arenas_.emplace(key, arena);
        return arena;
    }

    std::shared_ptr<MoEOverlayDispatchTicketStorage>
    Qwen35MoEGraph::dispatchTicketForForward(int layer_idx, int rows,
        DeviceId device, const DeviceMoEOverlayEpochTicket *epoch)
    {
        const auto intent = forward_input_construction_
            ? modelForwardInput().graph_submission_intent
            : ForwardGraphSubmissionIntent::Execute;
        if (intent != ForwardGraphSubmissionIntent::Execute &&
            intent != ForwardGraphSubmissionIntent::MaterializeExecutableWithoutLaunch &&
            intent != ForwardGraphSubmissionIntent::WorkspaceDeclaration)
            throw std::invalid_argument("Qwen35 dispatch ticket has an unknown original construction intent");
        auto storage = std::make_shared<MoEOverlayDispatchTicketStorage>();
        storage->bindForGraphConstruction(intent, layer_idx, rows,
            config_.moe.top_k, config_.d_model, device,
            /*workspace_generation=*/1,
            intent == ForwardGraphSubmissionIntent::WorkspaceDeclaration
                ? nullptr : mappedOverlayTicketArenaForDevice(device), epoch);
        return storage;
    }

    std::shared_ptr<MappedHostTransferArena>
    Qwen35MoEGraph::mappedOverlayTicketArenaForDevice(DeviceId device)
    {
        if (!device.is_gpu())
        {
            throw std::invalid_argument(
                "Qwen35 MoE mapped overlay ticket arena requires an exact GPU device");
        }
        const std::string key = device.to_string();
        const auto existing = moe_mapped_ticket_arenas_.find(key);
        if (existing != moe_mapped_ticket_arenas_.end())
        {
            if (!existing->second ||
                existing->second->devices().size() != 1u ||
                existing->second->devices().front() != device)
            {
                throw std::logic_error(
                    "Qwen35 MoE mapped ticket arena endpoint identity changed for " +
                    key);
            }
            return existing->second;
        }

        const std::array<DeviceId, 1> endpoints{device};
        auto arena = TransferEngine::instance().createMappedHostArena(endpoints);
        if (!arena)
        {
            throw std::runtime_error(
                "Qwen35 MoE could not create its mapped overlay ticket arena for " +
                key);
        }
        moe_mapped_ticket_arenas_.emplace(key, arena);
        return arena;
    }

    std::shared_ptr<MoEOverlayCollectiveWorkspace>
    Qwen35MoEGraph::overlayProtocolWorkspaceForParticipant(
        DeviceId graph_device,
        int participant,
        MoEOverlayReturnLayout return_layout)
    {
        if (!graph_device.is_valid() || participant < 0 ||
            config_.d_model <= 0 || config_.moe.top_k <= 0)
        {
            throw std::invalid_argument(
                "Qwen35 MoE overlay protocol workspace requires a valid graph device, participant, and geometry");
        }

        const size_t planned_rows = static_cast<size_t>(
            overlaySparseProtocolRowCapacity(config_, graph_device));
        const size_t top_k = static_cast<size_t>(config_.moe.top_k);
        if (planned_rows > std::numeric_limits<size_t>::max() / top_k)
        {
            throw std::overflow_error(
                "Qwen35 MoE overlay protocol entry capacity overflows size_t");
        }
        const size_t planned_entries = planned_rows * top_k;
        const std::string key =
            graph_device.to_string() + "#overlay_protocol_p" +
            std::to_string(participant) + "#return" +
            std::to_string(static_cast<uint32_t>(return_layout));
        const auto existing =
            moe_serial_overlay_protocol_workspaces_.find(key);
        if (existing != moe_serial_overlay_protocol_workspaces_.end())
        {
            if (!existing->second)
            {
                throw std::logic_error(
                    "Qwen35 MoE overlay protocol workspace cache contains a null arena");
            }
            existing->second->ensureCapacity(
                planned_rows,
                planned_entries,
                config_.d_model,
                config_.moe.top_k,
                DeviceId::cpu());
            return existing->second;
        }

        auto workspace = std::make_shared<MoEOverlayCollectiveWorkspace>(
            MoEOverlayCollectiveWorkspace::FixedCapacityConfig{
                .max_rows = planned_rows,
                .max_entries = planned_entries,
                .d_model = config_.d_model,
                .top_k = config_.moe.top_k,
                .device = DeviceId::cpu(),
                .reuse_policy = MoEOverlayCollectiveWorkspace::
                    StorageReusePolicy::SerialGraphFamily,
                .return_layout = return_layout,
            });
        moe_serial_overlay_protocol_workspaces_.emplace(key, workspace);
        return workspace;
    }

    std::shared_ptr<IMoEOverlayRankBatchTransport>
    Qwen35MoEGraph::overlayRankBatchTransportForGroup(
        DeviceId graph_device,
        int tier_index,
        int domain_ordinal,
        int source_world_rank,
        int target_world_rank,
        std::vector<int> participant_ids,
        const MoEExpertOwnerMap &owner_map,
        int source_participant_id)
    {
        if (!config_.moe.overlay_mpi_ctx || !graph_device.is_valid() ||
            tier_index < 0 || domain_ordinal < 0 ||
            source_world_rank < 0 || target_world_rank < 0 ||
            participant_ids.empty() || source_participant_id < 0 ||
            !model_ctx_ || !config_.moe.routed_expert_plan)
        {
            throw std::invalid_argument(
                "Qwen35 MoE rank-batch transport requires a complete remote-rank topology");
        }
        std::sort(participant_ids.begin(), participant_ids.end());
        if (participant_ids.front() < 0 ||
            std::adjacent_find(
                participant_ids.begin(), participant_ids.end()) !=
                participant_ids.end())
        {
            throw std::invalid_argument(
                "Qwen35 MoE rank-batch participant ids must be unique and non-negative");
        }

        const size_t planned_rows = static_cast<size_t>(
            overlaySparseProtocolRowCapacity(config_, graph_device));
        const size_t top_k = static_cast<size_t>(config_.moe.top_k);
        const size_t participant_count = participant_ids.size();
        const size_t row_multiplier = std::min(top_k, participant_count);
        if (planned_rows > std::numeric_limits<size_t>::max() / top_k ||
            planned_rows >
                std::numeric_limits<size_t>::max() / row_multiplier)
        {
            throw std::overflow_error(
                "Qwen35 MoE rank-batch aggregate capacity overflows size_t");
        }
        const size_t max_total_rows = planned_rows * row_multiplier;
        const size_t max_total_entries = planned_rows * top_k;

        const auto &overlay_plan = *config_.moe.routed_expert_plan;
        if (static_cast<size_t>(tier_index) >=
            overlay_plan.routed_tiers.size())
        {
            throw std::out_of_range(
                "Qwen35 MoE activation channel tier is outside the frozen plan");
        }
        const auto *const source_descriptor =
            owner_map.participantForId(source_participant_id);
        if (!source_descriptor || !source_descriptor->world_rank_known ||
            source_descriptor->world_rank != source_world_rank ||
            source_descriptor->tier_idx < 0 ||
            static_cast<size_t>(source_descriptor->tier_idx) >=
                overlay_plan.routed_tiers.size())
        {
            throw std::logic_error(
                "Qwen35 MoE activation channel cannot resolve its continuation endpoint");
        }
        const auto resolve_domain_ordinal =
            [&](const std::string &domain_name)
        {
            for (size_t index = 0;
                 index < overlay_plan.domains.size();
                 ++index)
            {
                if (overlay_plan.domains[index].name == domain_name)
                    return static_cast<int>(index);
            }
            throw std::logic_error(
                "Qwen35 MoE activation channel domain is absent from the frozen plan");
        };
        if (resolve_domain_ordinal(
                overlay_plan.routed_tiers[static_cast<size_t>(tier_index)]
                    .domain) != domain_ordinal)
        {
            throw std::logic_error(
                "Qwen35 MoE activation channel target domain ordinal diverged from its tier");
        }
        for (const int participant_id : participant_ids)
        {
            const auto *const participant =
                owner_map.participantForId(participant_id);
            if (!participant || !participant->world_rank_known ||
                participant->world_rank != target_world_rank ||
                participant->tier_idx != tier_index ||
                resolve_domain_ordinal(participant->domain_name) !=
                    domain_ordinal)
            {
                throw std::logic_error(
                    "Qwen35 MoE activation channel target group diverged from the frozen owner map");
            }
        }

        const bool retains_mtp_capacity =
            retainsMTPGraphCapacity(config_.mtp);
        const int max_decode_rows = retains_mtp_capacity
                                        ? std::max(
                                              1,
                                              resolveMTPRetainedTargetQueryRows(
                                                  config_.mtp))
                                        : 1;
        const int max_mtp_draft_depth =
            resolveMTPRetainedDraftCapacity(config_.mtp);
        const auto graph_family =
            resolveMoEOverlayInferenceGraphFamilyIdentity(
                model_ctx_->concreteLoader(),
                model_ctx_->architecture(),
                model_ctx_->totalBlockCount(),
                retains_mtp_capacity
                    ? MoEOverlayMTPGraphFamilyPolicy::RetainModelSidecars
                    : MoEOverlayMTPGraphFamilyPolicy::MainOnly,
                /*graph_family_generation=*/1,
                static_cast<int>(planned_rows),
                max_decode_rows,
                config_.max_request_count,
                max_mtp_draft_depth).forRoutedLayerInterval(
                    overlay_plan.first_model_layer,
                    overlay_plan.placementLayerCapacity());
        const bool same_rank = source_world_rank == target_world_rank;
        const auto transaction_topology = same_rank ?
            makeMoEOverlayRankLocalInferenceTopologyIdentity(owner_map, graph_family,
                source_participant_id, participant_ids) :
            makeMoEOverlayInferenceTopologyIdentity(
                owner_map,
                graph_family,
                source_world_rank,
                target_world_rank);
        const auto activation_graph_families =
            makeMoEOverlayActivationGraphFamilyManifests(graph_family);
        const MoEOverlayActivationLaneEndpoint source_endpoint{
            .world_rank = source_world_rank,
            .participant_id = source_participant_id,
            .tier_priority =
                overlay_plan
                    .routed_tiers[static_cast<size_t>(
                        source_descriptor->tier_idx)]
                    .priority,
            .domain_ordinal =
                resolve_domain_ordinal(source_descriptor->domain_name),
        };

        std::ostringstream key_stream;
        key_stream << graph_device.to_string()
                   << "#layers" << graph_family.first_model_layer
                   << ':' << graph_family.routedLayerCapacity()
                   << "#tier" << tier_index
                   << "#domain" << domain_ordinal
                   << "#rank" << source_world_rank << "to"
                   << target_world_rank
                   << "#rows" << planned_rows << "#p";
        for (const int participant : participant_ids)
            key_stream << participant << ',';
        const std::string key = key_stream.str();
        const auto existing = moe_overlay_rank_batch_transports_.find(key);
        if (existing != moe_overlay_rank_batch_transports_.end())
        {
            if (!existing->second ||
                existing->second->participantIds() != participant_ids)
            {
                throw std::logic_error(
                    "Qwen35 MoE rank-batch transport cache topology changed");
            }
            return existing->second;
        }

        const std::string channel_identity = same_rank ?
            makeMoEOverlayRankLocalGPUChannelIdentity(tier_index, domain_ordinal,
                source_world_rank, participant_ids) :
            makeMoEOverlayRankBatchChannelIdentity(
                tier_index,
                domain_ordinal,
                source_world_rank,
                target_world_rank,
                participant_ids);
        if (same_rank || resolveMoEOverlayRankBatchTransportKind(
                *config_.moe.overlay_mpi_ctx,
                source_world_rank,
                target_world_rank) ==
            MoEOverlayRankBatchTransportKind::NodeLocalSharedRows)
        {
            if (!config_.moe.rank_batch_transport_registry)
            {
                throw std::logic_error(
                    "Qwen35 MoE node-local activation graph has no preflight transport registry for " +
                    channel_identity);
            }
            auto transport =
                config_.moe.rank_batch_transport_registry->require(
                    channel_identity,
                    source_world_rank,
                    target_world_rank,
                    participant_ids);
            moe_overlay_rank_batch_transports_.emplace(key, transport);
            return transport;
        }

        auto wire_workspace =
            std::make_shared<MoEOverlayRankBatchWireWorkspace>(
                MoEOverlayRankBatchWireWorkspace::Config{
                    .participant_ids = participant_ids,
                    .max_total_rows = max_total_rows,
                    .max_total_entries = max_total_entries,
                    .d_model = config_.d_model,
                    .top_k = config_.moe.top_k,
                    .return_layout = source_descriptor->device.is_cpu()
                        ? MoEOverlayReturnLayout::CanonicalExpertRoutes
                        : MoEOverlayReturnLayout::ParticipantTokenPartials,
                });
        auto transport = createMoEOverlayRankBatchTransport(
            MoEOverlayRankBatchTransportConfig{
                .mpi_ctx = config_.moe.overlay_mpi_ctx,
                .physical_memory_authority = prepared_weight_store_->physicalMemoryAuthority(),
                .source_world_rank = source_world_rank,
                .target_world_rank = target_world_rank,
                .workspace = std::move(wire_workspace),
                .max_rows_per_participant = planned_rows,
                .max_entries_per_participant = max_total_entries,
                .d_model = config_.d_model,
                .top_k = config_.moe.top_k,
                .tier_index = tier_index,
                .domain_ordinal = domain_ordinal,
                .channel_identity = channel_identity,
                .transaction_slot_count = 4096,
                .transaction_topology = transaction_topology,
                .source_endpoint = source_endpoint,
                .target_tier_priority =
                    overlay_plan.routed_tiers[static_cast<size_t>(tier_index)]
                        .priority,
                .activation_graph_families = activation_graph_families,
                .local_lanes = [&]
                {
                    std::vector<MoEOverlayActivationLocalLaneBinding> lanes;
                    lanes.reserve(participant_ids.size());
                    for (const int participant_id : participant_ids)
                    {
                        lanes.push_back({
                            .participant_id = participant_id,
                            .device = graph_device,
                        });
                    }
                    return lanes;
                }(),
            });
        moe_overlay_rank_batch_transports_.emplace(key, transport);
        return transport;
    }

    DeviceMoERuntimeTable::Config Qwen35MoEGraph::moeRuntimeTableConfig(
        DeviceId device,
        const MoERuntimeTableIdentity &identity,
        int num_layers_override,
        bool bind_overlay_epoch) const
    {
        // Overrides are storage counts. The model-global origin comes only
        // from the authored PP stage, never from a currently executing layer.
        const int requested_layers = num_layers_override > 0
            ? num_layers_override : config_.n_layers;
        if (config_.pp_layer_offset < 0 || config_.n_layers <= 0 ||
            requested_layers < config_.n_layers ||
            requested_layers > std::numeric_limits<int>::max() - config_.pp_layer_offset)
            throw std::invalid_argument("Qwen35 MoE runtime table requires a valid complete stage interval");

        int table_layers = requested_layers;
        if (bind_overlay_epoch)
        {
            const auto runtime_plan = runtimePlanForGraph(config_);
            const auto &source_plan = runtime_plan
                ? runtime_plan->sourcePlanPtr() : config_.moe.routed_expert_plan;
            if (!source_plan || !source_plan->usesExpertOverlayAuthority())
                throw std::logic_error("Qwen35 MoE canonical overlay runtime table has no routed placement authority");
            if (source_plan->first_model_layer != config_.pp_layer_offset)
                throw std::invalid_argument("Qwen35 MoE runtime table placement belongs to a different pipeline stage");
            if (identity.role == MoERuntimeTableRole::MainDecodeDurablePlacement)
            {
                // The parent retains the entire stage manifest, including
                // terminal NextN banks, regardless of graph construction order.
                table_layers = source_plan->placementLayerCapacity(requested_layers);
            }
            else if (source_plan->placementLayerCapacity(config_.n_layers) < requested_layers)
                throw std::invalid_argument("Qwen35 MoE child runtime exceeds its stage placement manifest");
        }
        DeviceMoERuntimeTable::Config result{.device_id = device, .num_layers = table_layers,
            .num_experts = config_.moe.num_experts, .top_k = config_.moe.top_k,
            .mirror_to_device = true, .first_model_layer = config_.pp_layer_offset};
        if (intermediateShardArena())
        {
            const auto residency = config_.moe.expert_overlay_participant_residency;
            const auto manager = model_ctx_ ? model_ctx_->concreteWeightManager() : nullptr;
            if (!residency || !manager || !prepared_weight_store_)
                throw std::logic_error("GPU F-shard runtime requires its original model/registry/source family");
            const auto &owners = residency->initialOwnerMap();
            const auto found = std::find_if(owners.participants().begin(), owners.participants().end(), [&](const auto &member) {
                return member.device == device && member.domain_participant_index == config_.tp_device_idx &&
                    member.domain_name == config_.moe.routed_expert_plan->continuation_domain;
            });
            if (found == owners.participants().end())
                throw std::invalid_argument("GPU F-shard runtime has no exact original graph participant");
            for (int layer = result.first_model_layer; layer < result.first_model_layer + table_layers; ++layer)
            {
                const auto declaration = adoptMoEOverlayJointGraphDeclaration(*residency, manager->expertGemmRegistry(),
                    prepared_weight_store_->modelId(), found->participant_id, layer, config_.moe.num_experts,
                    config_.d_model, config_.moe.intermediate_size);
                result.fragment_families.push_back(declaration->family());
            }
        }
        return result;
    }

    IMoERuntimeTable *Qwen35MoEGraph::moeRuntimeTableForDevice(
        DeviceId device,
        const MoERuntimeTableIdentity &identity,
        int prefill_token_capacity,
        int num_layers_override,
        MoERuntimeHistogramProducerRole histogram_producer_role,
        bool bind_overlay_epoch)
    {
#if !defined(HAVE_ROCM) && !defined(HAVE_CUDA)
        (void)device;
        (void)identity;
        (void)prefill_token_capacity;
        (void)num_layers_override;
        (void)histogram_producer_role;
        (void)bind_overlay_epoch;
        return nullptr;
#else
        // Device-routed grouped MoE (decode + grouped prefill) is supported on any
        // GPU backend: MoERuntimeTable mirrors its placement banks through the
        // generic IBackend abstraction (see MoERuntimeTable.cpp mirrorBackend),
        // so CUDA and ROCm are both valid here. Gating on is_rocm() previously
        // left CUDA without a runtime table, which forced every MoE routing/expert
        // decode stage into a non-capturable manual graph segment.
        if (!device.is_gpu() || config_.moe.num_experts <= 0 || config_.moe.top_k <= 0)
            return nullptr;
        auto table_config = moeRuntimeTableConfig(device, identity, num_layers_override, bind_overlay_epoch);
        const int table_layers = table_config.num_layers;

        /*
         * Graph executables freeze direct route-scratch pointers. Allocate the
         * complete serial execution domain before the first graph is built:
         * ordinary prefill needs the largest configured graph bucket, while
         * grouped verification and request-batched MTP need their flattened
         * target-row capacity. Every role is event/stream ordered by the
         * orchestrator, so one largest-participant arena is both correct and
         * substantially smaller than one allocation per layer and MTP depth.
         */
        const int prefill_graph_rows = overlaySparseProtocolRowCapacity(
            config_, device);
        const int verifier_rows =
            retainsMTPGraphCapacity(config_.mtp)
                ? resolveMTPRetainedTargetQueryRows(config_.mtp)
                : 1;
        const int planned_route_rows =
            std::max({
                1,
                prefill_token_capacity,
                prefill_graph_rows,
                verifier_rows,
            });
        const std::string scratch_key = device.to_string();
        auto scratch_it =
            moe_serial_route_scratch_arenas_.find(scratch_key);
        if (scratch_it == moe_serial_route_scratch_arenas_.end())
        {
            DeviceMoESerialRouteScratchArena::Config scratch_config;
            scratch_config.device_id = device;
            scratch_config.num_experts = config_.moe.num_experts;
            scratch_config.top_k = config_.moe.top_k;
            scratch_config.token_capacity = planned_route_rows;
            scratch_it = moe_serial_route_scratch_arenas_
                             .emplace(
                                 scratch_key,
                                 std::make_shared<
                                     DeviceMoESerialRouteScratchArena>(
                                     scratch_config))
                             .first;
        }
        else if (!scratch_it->second ||
                 scratch_it->second->tokenCapacity() < planned_route_rows)
        {
            throw std::logic_error(
                "Qwen35 MoE graph requested route scratch beyond the immutable "
                "per-device plan for " +
                device.to_string() +
                ": requested=" + std::to_string(planned_route_rows) +
                " planned=" +
                std::to_string(
                    scratch_it->second
                        ? scratch_it->second->tokenCapacity()
                        : 0));
        }

        const std::string key = moeRuntimeTableKey(device, identity);
        DeviceMoERuntimeTable *overlay_placement_source = nullptr;
        if (bind_overlay_epoch)
        {
            if (identity.role == MoERuntimeTableRole::MTPDepth ||
                identity.role ==
                    MoERuntimeTableRole::CurrentBatchLLEPPrefill)
            {
                const MoERuntimeTableIdentity main_identity{
                    .role = MoERuntimeTableRole::MainDecodeDurablePlacement,
                    .mtp_depth = -1,
                };
                const std::string main_key =
                    moeRuntimeTableKey(device, main_identity);
                auto main_it = moe_runtime_tables_.find(main_key);
                if (main_it == moe_runtime_tables_.end() || !main_it->second)
                {
                    /*
                     * Prefill buckets may be captured before the first decode
                     * graph. Materialize the canonical table as topology now;
                     * the normal per-layer prepared-weight publication below
                     * initializes both this parent and the child. This is not
                     * an eager fallback or a second placement recipe: it makes
                     * graph construction order irrelevant while retaining one
                     * model-lifetime bank owner.
                     */
                    (void)moeRuntimeTableForDevice(
                        device,
                        main_identity,
                        prefill_token_capacity,
                        table_layers,
                        MoERuntimeHistogramProducerRole::NotProducer,
                        /*bind_overlay_epoch=*/true);
                    main_it = moe_runtime_tables_.find(main_key);
                    if (main_it == moe_runtime_tables_.end() ||
                        !main_it->second)
                    {
                        throw std::logic_error(
                            "Qwen35 MoE failed to materialize canonical "
                            "ExpertOverlay runtime table " +
                            main_key + " for a child graph");
                    }
                }
                overlay_placement_source = main_it->second.get();
            }
        }
        auto it = moe_runtime_tables_.find(key);
        if (it != moe_runtime_tables_.end())
        {
            if (!it->second || it->second->firstModelLayer() != table_config.first_model_layer ||
                it->second->layerCount() != table_layers)
                throw std::logic_error("Qwen35 MoE runtime table " + key + " changed its immutable stage interval");
            if (it->second->usesOverlayEpochTicket() != bind_overlay_epoch)
            {
                throw std::logic_error(
                    "Qwen35 MoE runtime table " + key +
                    " was reused with a different ExpertOverlay epoch-binding policy");
            }
            if (bind_overlay_epoch)
            {
                const auto binding =
                    deviceMoEOverlayEpochExecutionBinding(device);
                if (!binding ||
                    it->second->overlayEpochTicket() !=
                        binding.arena->requestTicket(binding.request_slot))
                {
                    throw std::logic_error(
                        "Qwen35 MoE runtime table " + key +
                        " does not share the canonical ExpertOverlay request ticket");
                }
            }
            if (it->second->overlayPlacementSource() !=
                overlay_placement_source)
            {
                throw std::logic_error(
                    "Qwen35 MoE runtime table " + key +
                    " was reused with a different canonical overlay placement authority");
            }
            if (prefill_token_capacity > 0)
                it->second->ensurePrefillRouteScratchCapacity(prefill_token_capacity);
            registerRuntimeTableHistogramSyncIfNeeded(
                key,
                it->second.get(),
                histogram_producer_role);
            if (identity.role == MoERuntimeTableRole::MTPDepth)
            {
                PerfStatsCollector::addCounter(
                    "mtp",
                    "moe_mtp_sidecar_runtime_table_reuses",
                    1.0,
                    "graph",
                    device.to_string(),
                    {{"runtime_table", key},
                     {"layers", std::to_string(table_layers)},
                     {"histogram_sync", "disabled"}});
            }
            return it->second.get();
        }

        const auto &authority = config_.moe.expert_overlay_residency_authority;
        const bool stationary_shards = authority && authority->stationaryTensorShardResidency();
        const bool mapped_ticket_histograms = intermediateShardArena() &&
            config_.moe.routed_expert_plan &&
            hasMappedGPUCPUIntermediateShardTopology(*config_.moe.routed_expert_plan);
        table_config.grouped_verifier_histogram_publication =
            identity.role ==
                    MoERuntimeTableRole::MainDecodeDurablePlacement &&
                    config_.moe.rebalance_config.mode !=
                        MoERebalanceRuntimeMode::Off &&
                    !stationary_shards && !mapped_ticket_histograms
                ? GroupedVerifierHistogramPublicationMode::AcceptedRows
                : GroupedVerifierHistogramPublicationMode::Disabled;
        if (bind_overlay_epoch && !stationary_shards &&
            config_.moe.rebalance_config.mode ==
                MoERebalanceRuntimeMode::Dynamic)
        {
            if (!authority)
            {
                throw std::logic_error(
                    "Qwen35 MoE Dynamic runtime table has no ExpertOverlay residency authority");
            }
            const auto &catalog =
                config_.moe.expert_overlay_residency_authority
                    ->economyLayerCatalog();
            if (!catalog)
            {
                throw std::logic_error(
                    "Qwen35 MoE Dynamic runtime table has no canonical economy layer catalog");
            }
            table_config.overlay_service_telemetry_coverage =
                MoEOverlayServiceTelemetryCoverage::
                    CatalogStratifiedSample;
            table_config.overlay_service_telemetry_catalog = catalog;
        }
        table_config.prefill_token_capacity = planned_route_rows;
        table_config.deferred_verifier_token_capacity =
            identity.role == MoERuntimeTableRole::MainDecodeDurablePlacement &&
                    retainsMTPGraphCapacity(config_.mtp)
                ? verifier_rows
                : 0;
        table_config.serial_route_scratch_arena = scratch_it->second;
        if (projection_arena_)
        {
            const auto residency = config_.moe.expert_overlay_participant_residency;
            if (!residency || !residency->projectionPreparation() || !model_ctx_)
                throw std::logic_error("Projection runtime has no authenticated participant preparation");
            const auto runtime_plan = routedForwardRecipe(config_.default_device).runtime;
            const auto &owners = *routedForwardRecipe(device).owners;
            const int id = participantIdForTierDevice(owners, 0, device);
            const auto *participant = owners.participantForId(id);
            if (!participant) throw std::logic_error("Projection runtime cannot resolve its graph-local participant");
            const auto manager = model_ctx_->concreteWeightManager();
            if (!manager) throw std::logic_error("Projection runtime has no prepared engine registry");
            table_config.fixed_down_banks.reserve(table_layers);
            for (int layer = table_config.first_model_layer;
                 layer < table_config.first_model_layer + table_layers; ++layer)
            {
                // Child graph identities retain the parent's exact immutable
                // bank; only routing scratch belongs to the child lifecycle.
                table_config.fixed_down_banks.push_back(overlay_placement_source
                    ? overlay_placement_source->retainFixedDownProjectionBank(layer)
                    : MoEOverlayFixedDownProjectionBank::resolve(manager->expertGemmRegistry(), *participant, layer,
                        residency->projectionPreparation()->requireProjectionOwnershipForParticipant(*participant, layer)));
            }
        }
        if (bind_overlay_epoch)
        {
            const auto binding =
                deviceMoEOverlayEpochExecutionBinding(device);
            if (!binding)
            {
                throw std::logic_error(
                    "Qwen35 MoE overlay runtime table requested an empty epoch binding on " +
                    device.to_string());
            }
            table_config.overlay_epoch_arena = binding.arena;
            table_config.overlay_epoch_ticket_slot = binding.request_slot;
            table_config.overlay_placement_source =
                overlay_placement_source;
        }

        auto table = std::make_unique<MoERuntimeTable>(table_config);
        IMoERuntimeTable *ptr = table.get();
        registerRuntimeTableHistogramSyncIfNeeded(
            key,
            ptr,
            histogram_producer_role);
        moe_runtime_tables_.emplace(key, std::move(table));
        if (identity.role == MoERuntimeTableRole::MTPDepth)
        {
            PerfStatsCollector::addCounter(
                "mtp",
                "moe_mtp_sidecar_runtime_table_creations",
                1.0,
                "graph",
                device.to_string(),
                {{"runtime_table", key},
                 {"layers", std::to_string(table_layers)},
                 {"num_experts", std::to_string(config_.moe.num_experts)},
                 {"top_k", std::to_string(config_.moe.top_k)},
                 {"histogram_sync", histogram_producer_role ==
                                            MoERuntimeHistogramProducerRole::ProductionDecode
                                        ? "enabled"
                                        : "disabled"}});
        }
        return ptr;
#endif
    }

    DeviceMoEOverlayEpochArena::Config
    Qwen35MoEGraph::moeOverlayEpochArenaConfig(DeviceId device) const
    {
        const auto manager = model_ctx_ ? model_ctx_->concreteWeightManager() : nullptr;
        const auto memory = manager ? manager->physicalMemoryAuthority() : nullptr;
        if (!memory || !memory->contains(device))
            throw std::logic_error("Qwen MoE epoch storage requires its original admitted model authority for " + device.to_string());
        if (prepared_weight_store_ && prepared_weight_store_->physicalMemoryAuthority() != memory)
            throw std::logic_error("Qwen MoE epoch storage rejects a prepared store with a different original model authority");
        DeviceMoEOverlayEpochArena::Config result;
        result.device_id = device;
        result.memory_authority = memory;
        result.initial_epoch = 1u;
        result.initial_bank = 1u;
        result.request_slot_capacity = 1u;
        return result;
    }

    std::shared_ptr<PhysicalMemoryAuthority>
    Qwen35MoEGraph::moeActivationBatchMemoryAuthority(DeviceId device) const
    {
        if (!prepared_weight_store_)
            throw std::logic_error("Original activation Batch graph requires its model-owned prepared weight store");
        return moeOverlayEpochArenaConfig(device).memory_authority;
    }

    DeviceMoEOverlayEpochExecutionBinding
    Qwen35MoEGraph::deviceMoEOverlayEpochExecutionBinding(DeviceId device)
    {
#if !defined(HAVE_ROCM) && !defined(HAVE_CUDA)
        (void)device;
        return {};
#else
        const auto runtime_plan = routedForwardRecipe(config_.default_device).runtime;
        const auto &source_plan = runtime_plan
                                      ? runtime_plan->sourcePlanPtr()
                                      : config_.moe.routed_expert_plan;
        const bool overlay_requested =
            source_plan && source_plan->usesExpertOverlayAuthority();
        if (!overlay_requested)
            return {};
        if (!device.is_gpu())
        {
            throw std::invalid_argument(
                "Qwen35 MoE ExpertOverlay epoch binding requires a GPU participant, got " +
                device.to_string());
        }

        std::optional<MoEOverlayDeviceControllerParticipantBinding>
            retirement_readiness_controller;
        if (config_.moe.device_controller_fabric)
        {
            for (const int participant_id :
                 config_.moe.device_controller_fabric->localParticipantIds())
            {
                const auto participant =
                    config_.moe.device_controller_fabric
                        ->participantBinding(participant_id);
                if (participant.device != device)
                    continue;
                if (retirement_readiness_controller ||
                    !participant.valid() || !participant.controller ||
                    !participant.lifetime)
                {
                    throw std::logic_error(
                        "Qwen35 MoE device maps ambiguously or incompletely to the topology-wide admission fabric: " +
                        device.to_string());
                }
                retirement_readiness_controller = participant;
            }
            if (!retirement_readiness_controller)
            {
                throw std::logic_error(
                    "Qwen35 MoE device has no local topology-wide admission binding: " +
                    device.to_string());
            }
        }

        // Model/PMA admission precedes arena initialization. Prepared weight
        // handles arrive later and cannot be the source of this cold ownership.
        auto arena_config = moeOverlayEpochArenaConfig(device);
        const std::string key = device.to_string();
        auto existing = moe_overlay_epoch_arenas_.find(key);
        if (existing != moe_overlay_epoch_arenas_.end())
        {
            if (!existing->second ||
                existing->second->deviceId() != device ||
                existing->second->requestSlotCapacity() != 1u)
            {
                throw std::logic_error(
                    "Qwen35 MoE ExpertOverlay epoch arena changed identity for " + key);
            }
            return {
                .arena = existing->second,
                .request_slot = 0u,
                .retirement_readiness_controller =
                    retirement_readiness_controller,
            };
        }

        /*
         * Every runtime table begins with bank zero empty, prepares epoch one
         * into its inactive bank, and flips that table to bank one during model
         * graph construction.  Publishing the epoch selector with the same
         * initial bank makes a subsequently acquired ticket valid for every
         * main and MTP layer without a setup-only host shadow.
         */
        if (config_.moe.device_controller_fabric)
        {
            const auto &participant =
                *retirement_readiness_controller;
                arena_config.external_admission_epoch =
                    &participant.controller->admission_epoch;
                arena_config.external_admission_lifetime =
                    participant.lifetime;
                if (participant.inference_epoch_member)
                {
                    if (!participant.inference_epoch_record)
                    {
                        throw std::logic_error(
                            "Qwen35 MoE continuation participant has no transaction epoch barrier: " +
                            device.to_string());
                    }
                    arena_config.admission_barrier = {
                        .record = participant.inference_epoch_record,
                        .participant_id = static_cast<std::uint32_t>(
                            participant.participant_id),
                        .storage_bytes = participant.inference_epoch_storage_bytes,
                    };
                }
        }
        auto arena =
            std::make_shared<DeviceMoEOverlayEpochArena>(arena_config);
        moe_overlay_epoch_arenas_.emplace(key, arena);
        return {
            .arena = std::move(arena),
            .request_slot = 0u,
            .retirement_readiness_controller =
                retirement_readiness_controller,
        };
#endif
    }

    void Qwen35MoEGraph::registerRuntimeTableHistogramSyncIfNeeded(
        const std::string &key,
        IMoERuntimeTable *table,
        MoERuntimeHistogramProducerRole histogram_producer_role)
    {
        if (histogram_producer_role !=
                MoERuntimeHistogramProducerRole::ProductionDecode ||
            !table || !config_.moe.decode_histogram)
            return;

        // Overlay demand has exactly one producer: CPU routing or the
        // declared heterogeneous ticket boundary. A second drain of accepted
        // GPU marginals would duplicate rows and discard their batch identity.
        // Native all-GPU controllers retain their evidence on device.
        if (config_.moe.expert_overlay_residency_authority)
            return;

        auto *histogram = config_.moe.decode_histogram;
        const std::string sync_key =
            key + "@" + std::to_string(reinterpret_cast<std::uintptr_t>(histogram));
        if (!moe_runtime_histogram_sync_keys_.insert(sync_key).second)
            return;
        histogram->registerRuntimeHistogramSync(
            [table, histogram]()
            {
                void *stream = table->decodeHistogramProducerStream();
                if (!stream)
                    throw std::runtime_error(
                        "[Qwen35MoEGraph] runtime histogram sync requested before a decode producer stream was recorded");
                return table->syncDecodeHistogramToHost(*histogram, stream);
            });
        PerfStatsCollector::addCounter(
            "moe_rebalance", "runtime_histogram_sync_registrations",
            1.0, "graph", "", {{"key", key}, {"protocol", "synchronous_legacy"}});
    }

    // =========================================================================
    // Schema
    // =========================================================================

    GraphSchema Qwen35MoEGraph::getSchema() const
    {
        Qwen35MoESchemaFactory factory;
        GraphSchema schema = factory.createSchema();
        if (const auto *projection = projectionArena()) projection->replaceWholeExpertSchema(schema);
        if (const auto *fragments = intermediateShardArena()) fragments->bindSchema(schema);

        completeHybridSchema(schema);

        return schema;
    }

    // =========================================================================
    // Resolver Config (MoE buffer registration)
    // =========================================================================

    GraphResolverConfig Qwen35MoEGraph::getResolverConfig(int seq_len) const
    {
        const auto *projection = projectionArena();
        // Start with base Qwen35 resolver config (GDN buffers, etc.)
        GraphResolverConfig config = Qwen35Graph::getResolverConfig(seq_len);

        // Add MoE-specific custom formulas
        int expert_intermediate = config_.moe.intermediate_size;
        const int shared_intermediate =
            std::max(0, config_.moe.shared_intermediate_size);
        const int max_ffn_intermediate =
            std::max(expert_intermediate, shared_intermediate);
        int top_k = config_.moe.top_k;
        const int mtp_target_query_rows =
            retainsMTPGraphCapacity(config_.mtp)
                ? std::max(
                      1,
                      resolveMTPRetainedTargetQueryRows(config_.mtp))
                : 1;
        const size_t moe_activation_rows = static_cast<size_t>(
            std::max(std::max(1, seq_len), mtp_target_query_rows));

        config.custom_formulas["moe_top_k"] = static_cast<size_t>(top_k);
        const int local_tp_degree =
            config_.tp_ctx && config_.tp_ctx->isLocal()
                ? std::max(1, config_.tp_ctx->degree())
                : 0;
        config.custom_formulas["moe_canonical_publication_slots"] =
            static_cast<size_t>(top_k + std::max(1, local_tp_degree));
        config.custom_formulas["moe_expert_intermediate"] = static_cast<size_t>(expert_intermediate);
        config.custom_formulas["moe_ffn_intermediate_max"] =
            static_cast<size_t>(max_ffn_intermediate);
        /*
         * MTP sidecar graphs reuse the MoE BufferIds so that snapshot and
         * publication naming stay aligned with the main graph, but the sidecar
         * can execute up to `mtp_target_query_rows` verifier rows while normal
         * decode allocates `seq_len == 1`.  Reserve the larger row count for
         * MoE scratch/output tensors so a four-row verifier cannot overrun a
         * one-row main decode buffer.
         */
        config.custom_formulas["moe_activation_rows"] = moe_activation_rows;

        // Standard MoE role identities are shared with metadata admission by
        // BufferArena. Only projection-specific banks add resolver mappings.
        if (projection) projection->bindResolver(config);
        if (const auto *fragments = intermediateShardArena()) fragments->bindResolver(config);

        LOG_DEBUG("[Qwen35MoEGraph::getResolverConfig] MoE formulas: "
                  << "moe_top_k=" << top_k
                  << ", moe_canonical_publication_slots="
                  << (top_k + std::max(1, local_tp_degree))
                  << ", moe_expert_intermediate=" << expert_intermediate
                  << ", moe_ffn_intermediate_max=" << max_ffn_intermediate
                  << ", moe_activation_rows=" << moe_activation_rows);

        return config;
    }

    // =========================================================================
    // MoE FFN Graph Building
    // =========================================================================

    ComputeGraph Qwen35MoEGraph::buildFFNGraph(
        const LayerWeights &layer,
        ActivationBuffers &buffers,
        int layer_idx,
        int seq_len,
        int batch_size,
        DeviceId device,
        void *device_state_publication_stream,
        const int32_t *sequence_lengths_device,
        const int32_t *absolute_position_ids_device)
    {
        const auto identity = mtpGraphContextActive()
            ? LayerGraphSourceIdentity::inlinePredictor(layer_idx, mtpGraphDepthIndex())
            : mainAttentionIdentity(layer_idx, nullptr).source();
        return buildFFNGraphForLayer(layer, buffers, identity, seq_len, batch_size,
            device, device_state_publication_stream, sequence_lengths_device,
            absolute_position_ids_device, nullptr);
    }

    ComputeGraph Qwen35MoEGraph::buildIntermediateShardFFNGraphForLayer(
        const LayerWeights &layer, ActivationBuffers &buffers, const LayerGraphSourceIdentity &identity,
        int rows, DeviceId device, void *stream, const int32_t *count, const int32_t *positions,
        const Qwen4ExpPredictorResidualBinding *predictor, MoEOverlayServicePhaseHint phase)
    {
        const auto plan = config_.moe.routed_expert_plan;
        const auto residency = config_.moe.expert_overlay_participant_residency;
        const auto manager = model_ctx_ ? model_ctx_->concreteWeightManager() : nullptr;
        auto *tp = dynamic_cast<ILocalTPContext *>(config_.tp_ctx);
        const int layer_idx = identity.sourceLayer();
        const auto prefix = identity.prefix();
        const bool grouped = rows != 1 || phase != MoEOverlayServicePhaseHint::Decode;
        if (!device.is_gpu() || !stream || rows <= 0 || !plan || !residency || !manager || !tp ||
            !prepared_weight_store_ || !weight_bindings_.original_frozen_source ||
            phase == MoEOverlayServicePhaseHint::Auto)
            throw std::invalid_argument("GPU F-shard model FFN requires original source/store/family/native TP, explicit rows/phase/stream");
        // This graph owns only the local cohort's partial sum. Admission can
        // retain Dynamic intent if that complete cohort has no alternate tier.
        const auto &authority = config_.moe.expert_overlay_residency_authority;
        if ((config_.moe.rebalance_config.mode != MoERebalanceRuntimeMode::Off &&
             (config_.moe.rebalance_config.mode != MoERebalanceRuntimeMode::Dynamic ||
              !authority || (!authority->stationaryTensorShardResidency() && !authority->noPeerResidency()))) ||
            plan->routed_tiers.size() != 1 || plan->continuation_domain != plan->routed_tiers.front().domain ||
            plan->shared_expert_domain != plan->continuation_domain)
            throw std::invalid_argument("GPU F-shard local model FFN requires one original stationary routed/shared/continuation domain");
        const auto &owners = residency->initialOwnerMap();
        const MoEExpertOwnerParticipant *participant = nullptr;
        for (const auto &member : owners.participants())
            if (member.device == device && member.domain_name == plan->continuation_domain &&
                member.domain_participant_index == config_.tp_device_idx)
            {
                if (participant) throw std::invalid_argument("GPU F-shard model FFN has ambiguous original participant");
                participant = &member;
            }
        if (!participant) throw std::invalid_argument("GPU F-shard model FFN has no original participant for layer " + std::to_string(layer_idx));
        const auto declaration = adoptMoEOverlayJointGraphDeclaration(*residency, manager->expertGemmRegistry(),
            prepared_weight_store_->modelId(), participant->participant_id, layer_idx,
            config_.moe.num_experts, config_.d_model, config_.moe.intermediate_size);
        auto initial = declaration->acquireInitialConstruction();
        if (!initial) throw std::logic_error("GPU F-shard model FFN original construction source already retired");
        const auto &source = initial->layerForModelLayer(layer_idx);
        const bool mtp = identity.workload() == GraphLayerWorkload::MTPDepth;
        const MoERuntimeTableIdentity runtime_identity{
            .role = mtp ? MoERuntimeTableRole::MTPDepth : MoERuntimeTableRole::MainDecodeDurablePlacement,
            .mtp_depth = mtp ? identity.predictorDepth().value_or(-1) : -1};
        auto *runtime = dynamic_cast<DeviceMoERuntimeTable *>(moeRuntimeTableForDevice(device, runtime_identity,
            rows, std::max(config_.n_layers, layer_idx - config_.pp_layer_offset + 1),
            MoERuntimeHistogramProducerRole::NotProducer, true));
        if (!runtime) throw std::logic_error("GPU F-shard model FFN has no original mirrored runtime");
        publishInitialMoEIntermediateShardRuntimeLayer(*runtime, owners, source, layer_idx, stream);

        ComputeGraph graph;
        std::optional<Qwen4ExpHyperconnectionBuilder::BlockRead> residual_read;
        std::string norm;
        if (config_.flash_next)
        {
            residual_read.emplace(addBranchBlockRead(graph, prefix, buffers, layer_idx, rows, device,
                ResidualBlock::FeedForward, predictor));
            norm = residual_read->terminal();
        }
        else
        {
            FusedResidualNormStage::Params p;
            p.device_id = device; p.input = buffers.attn_proj; p.residual = buffers.current_hidden;
            p.gamma = layer.ffn_norm; p.norm_output = buffers.normalized; p.eps = config_.rms_norm_eps;
            p.seq_len = rows; p.hidden_dim = config_.d_model;
            p.input_buffer_id = buffers.idFor(BufferId::ATTN_PROJ);
            p.residual_buffer_id = buffers.idFor(BufferId::HIDDEN_STATE);
            p.norm_output_buffer_id = buffers.idFor(BufferId::NORMALIZED);
            norm = prefix + "ffn_norm";
            graph.addNode(norm, ComputeStageFactory::createFusedResidualNorm(p), device);
        }
        const auto owner = std::make_shared<MoERoutedPipelineKernelOwner>();
        const bool original_m1_hidden_q8=config_.flash_next&&config_.qfn_hidden_q8_reuse_candidate&&!mtp&&!grouped&&
            phase==MoEOverlayServicePhaseHint::Decode;
        if(original_m1_hidden_q8) {
            if(rows!=1||count||config_.qfn_resident_ffn_candidate||deviceMoEWeightFormatIsFloating(
                    MoEIntermediateShardDescriptorDeclaration::fromOriginalFamily(source.fragment_family).weightFormat())||
               !owner->router_q8_publication->requireOriginalM1Consumer(buffers.normalized,config_.d_model))
                throw std::invalid_argument("QFN hidden Q8 reuse requires original quantized fully-active ordinary M1 rows");
        }
        auto *indices = buffers.get(buffers.idFor(BufferId::MOE_EXPERT_INDICES));
        auto *weights = buffers.get(buffers.idFor(BufferId::MOE_EXPERT_WEIGHTS));
        MoERoutingStage::Params route;
        route.device_id = device; route.input = buffers.normalized; route.gate_weights = layer.moe_gate;
        route.seq_len = rows; route.d_model = config_.d_model; route.layer_idx = layer_idx;
        route.num_experts = config_.moe.num_experts; route.top_k = config_.moe.top_k;
        route.norm_topk_prob = config_.moe.norm_topk_prob;
        route.output_indices = indices; route.output_weights = weights;
        route.moe_runtime_table = runtime; route.routed_pipeline_kernel_owner = owner;
        route.routed_row_execution_policy = RoutedExpertRowExecutionPolicy::IntermediateShardLocal;
        // Static rebalance-Off graphs publish no histogram. Accepted-row
        // demand is wired only with the later moving-policy transaction.
        route.collect_device_runtime_histogram = false;
        route.active_row_count_device = count; route.absolute_position_ids_device = positions;
        route.force_decode_equivalent_verifier_prefill = grouped && phase != MoEOverlayServicePhaseHint::Prefill;
        if(!mtp && phase==MoEOverlayServicePhaseHint::GroupedVerifier)
            route.verifier_kernel_policy=plan->local_gpu_expert_kernel_policy;
        route.force_grouped_verifier_prefill_for_decode = grouped && rows == 1;
        // The local-F pipeline has projection stages instead of an expert-FFN
        // stage. Its router is the sole verifier-history boundary, including
        // the explicit publication-free contract of this static placement.
        route.grouped_verifier_histogram_role = selectMoEGroupedVerifierHistogramRole(
            route.force_decode_equivalent_verifier_prefill, /*collects_history=*/false);
        route.input_buffer_id = buffers.idFor(BufferId::NORMALIZED);
        route.output_indices_buffer_id = buffers.idFor(BufferId::MOE_EXPERT_INDICES);
        route.output_weights_buffer_id = buffers.idFor(BufferId::MOE_EXPERT_WEIGHTS);
        const auto router = prefix + "moe_routing";
        graph.addNode(router, ComputeStageFactory::createMoERouting(route), device);
        graph.addDependency(router, norm);
        MoEIntermediateShardPipelineParams local;
        local.family = declaration->family(); local.runtime = runtime; local.kernel_owner = owner;
        local.rows = rows; local.top_k = config_.moe.top_k; local.phase = phase;
        local.row_mode = grouped ? MoEIntermediateShardRowMode::Grouped : MoEIntermediateShardRowMode::Decode;
        local.verifier_kernel_policy=route.verifier_kernel_policy;
        // These independent diagnostic policies select only the original ROCm
        // ordinary-main native M1 transaction. Grouped verification, predictor,
        // shared/floating programs retain their declared Staged implementations.
        const bool original_main_m1=config_.flash_next&&!mtp&&!grouped&&
            phase==MoEOverlayServicePhaseHint::Decode;
        if(original_main_m1&&(config_.qfn_gate_up_completion_candidate||config_.qfn_down_completion_candidate))
        {
            if(!device.is_rocm()||rows!=1||config_.qfn_resident_ffn_candidate||deviceMoEWeightFormatIsFloating(
                MoEIntermediateShardDescriptorDeclaration::fromOriginalFamily(source.fragment_family).weightFormat()))
                throw std::invalid_argument("QFN completion requires the original native ROCm ordinary-main M1 program");
            if(config_.qfn_gate_up_completion_candidate)local.gate_up_publication=MoEGateUpPublicationPolicy::LastProducer;
            if(config_.qfn_down_completion_candidate)local.down_publication=MoEDownPublicationPolicy::LastProducerRouteFold;
        }
        if (count) local.live_rows = DeviceRowRange::deviceCounted(rows, count);
        local.prefix = prefix + "local_routed/"; local.router_node = router;
        local.hidden = {buffers.normalized, buffers.idFor(BufferId::NORMALIZED)};
        local.routing_indices = {indices, buffers.idFor(BufferId::MOE_EXPERT_INDICES)};
        local.routing_weights = {weights, buffers.idFor(BufferId::MOE_EXPERT_WEIGHTS)};
        local.route_partials = {buffers.get(buffers.idFor(BufferId::MOE_CANONICAL_ROUTE_CONTRIBUTIONS)),
            buffers.idFor(BufferId::MOE_CANONICAL_ROUTE_CONTRIBUTIONS)};
        local.output = {buffers.get(buffers.idFor(BufferId::MOE_COMBINED_OUTPUT)), buffers.idFor(BufferId::MOE_COMBINED_OUTPUT)};
        const auto routed = appendMoEIntermediateShardPipeline(graph, local, source);
        // Descriptor export has finished. No executable graph retains this
        // initial reader: the invocation acquires the device epoch itself.
        initial = {};

        // The selected fragment policy owns local shared F in both serial
        // and grouped graphs. The general dense accessor may merge a full-F
        // replicated decode/sidecar view, so bind this branch from its actual
        // original per-device frozen source, independently of that dense view.
        const auto &original_shared = *weight_bindings_.original_frozen_source;
        const auto *shared_gate = original_shared.optionalLayerForDevice(layer_idx, "ffn_gate_shexp.weight", device);
        const auto *shared_up = original_shared.optionalLayerForDevice(layer_idx, "ffn_up_shexp.weight", device);
        const auto *shared_down = original_shared.optionalLayerForDevice(layer_idx, "ffn_down_shexp.weight", device);
        SharedExpertFFNStage::Params shared;
        shared.device_id = device; shared.input = buffers.normalized;
        shared.gate_w = legacyTensor(shared_gate);
        shared.up_w = legacyTensor(shared_up);
        shared.down_w = legacyTensor(shared_down);
        if (!shared.gate_w || !shared.up_w || !shared.down_w)
            throw std::invalid_argument("GPU F-shard model FFN is missing original local shared triplet at layer " + std::to_string(layer_idx));
        shared.output = buffers.get(buffers.idFor(BufferId::MOE_SHARED_EXPERT_OUTPUT));
        shared.gate_scratch = buffers.get(buffers.idFor(BufferId::MOE_GATE_SCRATCH));
        shared.up_scratch = buffers.get(buffers.idFor(BufferId::MOE_UP_SCRATCH));
        shared.seq_len = rows; shared.d_model = config_.d_model; shared.intermediate = static_cast<int>(shared.gate_w->rows());
        shared.active_row_count_device = count;
        shared.output_role = SharedExpertOutputRole::ParticipantLocalPartial; shared.phase = phase;
        shared.force_grouped_verifier_prefill_for_decode = grouped && phase != MoEOverlayServicePhaseHint::Prefill;
        if (shared.force_grouped_verifier_prefill_for_decode) shared.required_router_q8_publication = owner->router_q8_publication;
        shared.prepared_store = prepared_weight_store_;
        shared.prepared_ref_gate = preparedRefForGraphWeight(shared_gate, device);
        shared.prepared_ref_up = preparedRefForGraphWeight(shared_up, device);
        shared.prepared_ref_down = preparedRefForGraphWeight(shared_down, device);
        shared.input_buffer_id = buffers.idFor(BufferId::NORMALIZED);
        shared.output_buffer_id = buffers.idFor(BufferId::MOE_SHARED_EXPERT_OUTPUT);
        shared.gate_scratch_buffer_id = buffers.idFor(BufferId::MOE_GATE_SCRATCH);
        shared.up_scratch_buffer_id = buffers.idFor(BufferId::MOE_UP_SCRATCH);
        if(original_m1_hidden_q8) {
            if(shared.input!=local.hidden.tensor||shared.input==local.route_partials.tensor||shared.input==local.output.tensor||
               shared.input==shared.output||shared.input==shared.gate_scratch||shared.input==shared.up_scratch)
                throw std::invalid_argument("QFN original M1 hidden publication requires one immutable shared source and distinct writable roles");
            shared.required_router_q8_publication=owner->router_q8_publication;
        }
        if (shared.force_grouped_verifier_prefill_for_decode||original_m1_hidden_q8)
        {
            DeviceMoEExpertDescriptor descriptor;
            if (!shared.prepared_ref_gate || !shared.prepared_ref_up || !shared.prepared_ref_down ||
                !exportDeviceMoEExpertWeightDescriptors(prepared_weight_store_->gemmKernel(*shared.prepared_ref_gate),
                    prepared_weight_store_->gemmKernel(*shared.prepared_ref_up), prepared_weight_store_->gemmKernel(*shared.prepared_ref_down),
                    config_.d_model, shared.intermediate, descriptor))
                throw std::invalid_argument("GPU F-shard model FFN has no supported original shared prepared descriptor");
            if(original_m1_hidden_q8&&deviceMoEWeightFormatIsFloating(descriptor.weight_format))
                throw std::invalid_argument("QFN hidden Q8 reuse has no floating shared projection quantizer to remove");
            if (!deviceMoEWeightFormatIsFloating(descriptor.weight_format) &&
                !shared.required_router_q8_publication->requireConsumerRows(rows, config_.d_model))
                throw std::invalid_argument("GPU F-shard model FFN original shared/router Q8 publication geometry disagrees");
        }
        const auto shared_node = prefix + "shared_expert_ffn";
        graph.addNode(shared_node, ComputeStageFactory::createSharedExpertFFN(shared), device);
        graph.addDependency(shared_node, norm);
        // Routed and shared stages borrow the admitted gate/up scratch in
        // sequence; finalization adds the router-publication dependency.
        graph.addDependency(shared_node, routed.producer);
        MoEIntermediateShardFFNFinalization finish;
        if(config_.flash_next && !mtp && !grouped && phase==MoEOverlayServicePhaseHint::Decode &&
            (config_.qfn_resident_ffn_candidate || qfnRetainedM1HCFFNEnabled(
                config_.moe.routed_expert_plan && config_.moe.routed_expert_plan->usesGFX906MI50Kernels() &&
                !hasMappedGPUCPUIntermediateShardTopology(*config_.moe.routed_expert_plan) &&
                config_.flash_next->hasRetainedM1HCFFNSources(layer_idx))))
            finish.execution=MoEIntermediateShardFFNExecution::ResidentCandidate;
        finish.shared_ffn_node = shared_node; finish.shared_weights = weight_bindings_.original_frozen_source;
        finish.shared_partition = config_.tp_config.get();
        finish.local_combined = {buffers.attn_proj, buffers.idFor(BufferId::ATTN_PROJ)};
        finish.prefix = prefix + "local_ffn/";
        finish.final_sum.device_id = device; finish.final_sum.tp_ctx = tp; finish.final_sum.tensor = buffers.attn_proj;
        finish.final_sum.count = std::size_t(rows) * config_.d_model;
        if (phase == MoEOverlayServicePhaseHint::Prefill)
        {
            const auto prefill_plan = resolveTPAllreducePlan(
                buffers.attn_proj,
                finish.final_sum.count,
                device,
                std::nullopt,
                ForwardExecutionPhase::Prefill);
            finish.final_sum.precision = prefill_plan.transport_precision;
            finish.final_sum.arithmetic_policy = prefill_plan.arithmetic_policy;
        }
        else
        {
            finish.final_sum.precision = "fp32";
            finish.final_sum.arithmetic_policy = TPAllreduceArithmeticPolicy::CanonicalRankOrder;
        }
        finish.final_sum.tensor_buffer_id = buffers.idFor(BufferId::ATTN_PROJ);
        finish.final_sum.sideband_device_index = config_.tp_device_idx;
        if (local.live_rows) finish.final_sum.live_rows = NativeCollectiveRows(*local.live_rows, config_.d_model);
        auto terminal = finalizeMoEIntermediateShardFFN(graph, routed, std::move(finish)).producer;
        if (residual_read)
            terminal = addBranchBlockWrite(graph, std::move(*residual_read), buffers, terminal,
                predictor ? &predictor->ffnOutput() : nullptr);
        else if (layer_idx >= config_.pp_layer_offset + config_.n_layers - 1)
        {
            ResidualAddStage::Params p;
            p.device_id = device; p.input = buffers.attn_proj; p.residual = buffers.current_hidden; p.output = buffers.current_hidden;
            p.num_elements = std::size_t(rows) * config_.d_model;
            p.input_buffer_id = buffers.idFor(BufferId::ATTN_PROJ);
            p.residual_buffer_id = p.output_buffer_id = buffers.idFor(BufferId::HIDDEN_STATE);
            const auto residual = prefix + "ffn_residual";
            graph.addNode(residual, ComputeStageFactory::createResidualAdd(p), device);
            graph.addDependency(residual, terminal); terminal = residual;
        }
        graph.setTerminalNode(terminal);
        return graph;
    }

    ComputeGraph Qwen35MoEGraph::buildCPUIntermediateShardFFNGraphForLayer(
        const LayerWeights &layer, ActivationBuffers &buffers, const LayerGraphSourceIdentity &identity,
        int rows, DeviceId device, const Qwen4ExpPredictorResidualBinding *predictor,
        MoEOverlayServicePhaseHint phase)
    {
        const auto plan = config_.moe.routed_expert_plan;
        const auto residency = config_.moe.expert_overlay_participant_residency;
        const auto manager = model_ctx_ ? model_ctx_->concreteWeightManager() : nullptr;
        auto *gtp = dynamic_cast<IGlobalTPContext *>(config_.tp_ctx);
        const int layer_idx = identity.sourceLayer();
        const auto prefix = identity.prefix();
        const bool mtp = identity.workload() == GraphLayerWorkload::MTPDepth;
        if (!device.is_cpu() || rows <= 0 || !plan || !residency || !manager || !gtp ||
            gtp->degree() < 2 || !prepared_weight_store_ || !layer.moe_gate ||
            phase == MoEOverlayServicePhaseHint::Auto)
        {
            throw std::invalid_argument(
                "CPU F-shard model FFN requires original CPU source/store/family/global TP and positive rows/phase");
        }
        const auto &authority = config_.moe.expert_overlay_residency_authority;
        if ((config_.moe.rebalance_config.mode != MoERebalanceRuntimeMode::Off &&
             (config_.moe.rebalance_config.mode != MoERebalanceRuntimeMode::Dynamic ||
              !authority || (!authority->stationaryTensorShardResidency() && !authority->noPeerResidency()))) ||
            plan->domains.size() != 1u || plan->routed_tiers.size() != 1u ||
            plan->continuation_domain != plan->routed_tiers.front().domain ||
            plan->shared_expert_domain != plan->continuation_domain)
        {
            throw std::invalid_argument(
                "CPU F-shard local model FFN requires one original stationary routed/shared/continuation CPU domain");
        }
        const auto &domain = plan->domains.front();
        if (domain.routed_compute_policy != RoutedExpertComputePolicy::TensorSharded ||
            domain.participants.size() != static_cast<size_t>(gtp->degree()) ||
            !std::all_of(domain.participants.begin(), domain.participants.end(),
                         [](const GlobalDeviceAddress &member) { return member.isCPU(); }))
        {
            throw std::invalid_argument(
                "CPU F-shard model FFN requires homogeneous CPU TensorSharded participants matching global TP degree");
        }

        const auto &owners = residency->initialOwnerMap();
        const int local_world_rank =
            config_.moe.overlay_mpi_ctx ? config_.moe.overlay_mpi_ctx->rank() : -1;
        const MoEExpertOwnerParticipant *participant = nullptr;
        for (const auto &member : owners.participants())
        {
            if (member.device == device &&
                member.domain_name == plan->continuation_domain &&
                member.domain_participant_index == config_.tp_device_idx &&
                (!member.world_rank_known || local_world_rank < 0 || member.world_rank == local_world_rank))
            {
                if (participant)
                {
                    throw std::invalid_argument(
                        "CPU F-shard model FFN has ambiguous original participant");
                }
                participant = &member;
            }
        }
        if (!participant)
        {
            throw std::invalid_argument(
                "CPU F-shard model FFN has no original participant for layer " + std::to_string(layer_idx));
        }

        const auto declaration = adoptMoEOverlayJointGraphDeclaration(
            *residency, manager->expertGemmRegistry(), prepared_weight_store_->modelId(),
            participant->participant_id, layer_idx,
            config_.moe.num_experts, config_.d_model, config_.moe.intermediate_size);
        const auto initial = declaration->acquireInitialConstruction();
        if (!initial)
        {
            throw std::logic_error(
                "CPU F-shard model FFN original construction source already retired");
        }
        const auto selection = declaration->sourceSelection();
        if (!selection)
        {
            throw std::logic_error(
                "CPU F-shard model FFN lost its original source selection");
        }
        const auto &source = initial->layerForModelLayer(layer_idx);
        const int local_intermediate =
            selection->ownership().projection(WeightRole::MoEExpertUp).rows;
        if (local_intermediate <= 0 ||
            source.fragments.size() != static_cast<size_t>(config_.moe.num_experts))
        {
            throw std::logic_error(
                "CPU F-shard model FFN has invalid local intermediate shard geometry for layer " +
                std::to_string(layer_idx));
        }

        std::vector<ITensorGemm *> prepared_gate(static_cast<size_t>(config_.moe.num_experts), nullptr);
        std::vector<ITensorGemm *> prepared_up(static_cast<size_t>(config_.moe.num_experts), nullptr);
        std::vector<ITensorGemm *> prepared_down(static_cast<size_t>(config_.moe.num_experts), nullptr);
        std::vector<std::shared_ptr<ITensorGemm>> owned_kernels;
        owned_kernels.reserve(static_cast<size_t>(config_.moe.num_experts) * 3u);
        for (int expert = 0; expert < config_.moe.num_experts; ++expert)
        {
            const auto &triplet = source.fragments[static_cast<size_t>(expert)];
            if (!selection->contains(expert) || !triplet.valid() ||
                triplet.gate()->get_n() != local_intermediate || triplet.gate()->get_k() != config_.d_model ||
                triplet.up()->get_n() != local_intermediate || triplet.up()->get_k() != config_.d_model ||
                triplet.down()->get_n() != config_.d_model || triplet.down()->get_k() != local_intermediate)
            {
                throw std::logic_error(
                    "CPU F-shard model FFN requires complete local shard triplets for every expert in layer " +
                    std::to_string(layer_idx) + " (expert=" + std::to_string(expert) + ")");
            }
            prepared_gate[static_cast<size_t>(expert)] = triplet.gate().get();
            prepared_up[static_cast<size_t>(expert)] = triplet.up().get();
            prepared_down[static_cast<size_t>(expert)] = triplet.down().get();
            owned_kernels.push_back(triplet.gate());
            owned_kernels.push_back(triplet.up());
            owned_kernels.push_back(triplet.down());
        }

        const bool force_verifier =
            !mtp && config_.grouped_mtp_verifier && config_.compute_all_position_logits;
        const LayerWeightBindings layer_bindings =
            predictor ? predictor->weights().fa_block : layerWeightBindingsForGraph(layer_idx);

        ComputeGraph graph;
        std::optional<Qwen4ExpHyperconnectionBuilder::BlockRead> residual_read;
        std::string norm_node;
        if (config_.flash_next)
        {
            residual_read.emplace(addBranchBlockRead(
                graph, prefix, buffers, layer_idx, rows, device, ResidualBlock::FeedForward, predictor));
            norm_node = residual_read->terminal();
        }
        else
        {
            FusedResidualNormStage::Params norm_params;
            norm_params.device_id = device;
            norm_params.input = buffers.attn_proj;
            norm_params.residual = buffers.current_hidden;
            norm_params.gamma = layer.ffn_norm;
            norm_params.norm_output = buffers.normalized;
            norm_params.eps = config_.rms_norm_eps;
            norm_params.seq_len = rows;
            norm_params.hidden_dim = config_.d_model;
            norm_params.input_buffer_id = buffers.idFor(BufferId::ATTN_PROJ);
            norm_params.residual_buffer_id = buffers.idFor(BufferId::HIDDEN_STATE);
            norm_params.norm_output_buffer_id = buffers.idFor(BufferId::NORMALIZED);
            norm_node = prefix + "ffn_norm";
            graph.addNode(norm_node, ComputeStageFactory::createFusedResidualNorm(norm_params), device);
        }

        // Routing and routed-expert execution share canonical CPU Q8_1 hidden publication
        // directly through one graph-local kernel owner (no sparse overlay dispatch/return).
        const auto routed_kernel_owner = std::make_shared<MoERoutedPipelineKernelOwner>();
        TensorBase *indices = buffers.get(buffers.idFor(BufferId::MOE_EXPERT_INDICES));
        TensorBase *weights = buffers.get(buffers.idFor(BufferId::MOE_EXPERT_WEIGHTS));
        TensorBase *moe_output = buffers.get(buffers.idFor(BufferId::MOE_COMBINED_OUTPUT));
        TensorBase *shared_output = buffers.get(buffers.idFor(BufferId::MOE_SHARED_EXPERT_OUTPUT));

        MoERoutingStage::Params route_params;
        route_params.device_id = device;
        route_params.input = buffers.normalized;
        route_params.gate_weights = layer.moe_gate;
        route_params.seq_len = rows;
        route_params.d_model = config_.d_model;
        route_params.layer_idx = layer_idx;
        route_params.num_experts = config_.moe.num_experts;
        route_params.top_k = config_.moe.top_k;
        route_params.norm_topk_prob = config_.moe.norm_topk_prob;
        route_params.output_indices = indices;
        route_params.output_weights = weights;
        route_params.routed_pipeline_kernel_owner = routed_kernel_owner;
        route_params.routed_row_execution_policy = RoutedExpertRowExecutionPolicy::IntermediateShardLocal;
        route_params.collect_device_runtime_histogram = false;
        route_params.force_decode_equivalent_verifier_prefill = force_verifier;
        route_params.input_buffer_id = buffers.idFor(BufferId::NORMALIZED);
        route_params.output_indices_buffer_id = buffers.idFor(BufferId::MOE_EXPERT_INDICES);
        route_params.output_weights_buffer_id = buffers.idFor(BufferId::MOE_EXPERT_WEIGHTS);
        const std::string router_node = prefix + "moe_routing";
        graph.addNode(router_node, ComputeStageFactory::createMoERouting(route_params), device);
        graph.addDependency(router_node, norm_node);

        const bool has_shared_expert =
            layer.shared_expert_gate && layer.shared_expert_up &&
            layer.shared_expert_down && shared_output;
        const bool shared_requires_allreduce =
            has_shared_expert && needsTPAllreduce() &&
            (predictor ? predictor->layout().sharedNeedsReduction() : denseTPAllreduceEnabledForCurrentGraph());

        std::string shared_node;
        if (has_shared_expert)
        {
            SharedExpertFFNStage::Params cpu_shared_params;
            cpu_shared_params.device_id = device;
            cpu_shared_params.input = buffers.normalized;
            cpu_shared_params.gate_w = layer.shared_expert_gate;
            cpu_shared_params.up_w = layer.shared_expert_up;
            cpu_shared_params.down_w = layer.shared_expert_down;
            cpu_shared_params.output = shared_output;
            cpu_shared_params.gate_scratch = buffers.get(buffers.idFor(BufferId::MOE_GATE_SCRATCH));
            cpu_shared_params.up_scratch = buffers.get(buffers.idFor(BufferId::MOE_UP_SCRATCH));
            cpu_shared_params.seq_len = rows;
            cpu_shared_params.d_model = config_.d_model;
            cpu_shared_params.intermediate = static_cast<int>(layer.shared_expert_gate->rows());
            cpu_shared_params.force_decode_equivalent_verifier_prefill = force_verifier;
            cpu_shared_params.disable_grouped_decode_shortcut =
                mtp && (predictor
                            ? predictor->layout().policy() == MTPShiftedKVHeadLayout::FullModelPerParticipant
                            : (config_.dense_tp_decode_replicated || config_.mtpUsesReplicatedDenseSidecarBinding()));
            cpu_shared_params.prepared_store = prepared_weight_store_;
            cpu_shared_params.prepared_ref_gate = preparedRefForGraphWeight(layer_bindings.shared_expert_gate, device);
            cpu_shared_params.prepared_ref_up = preparedRefForGraphWeight(layer_bindings.shared_expert_up, device);
            cpu_shared_params.prepared_ref_down = preparedRefForGraphWeight(layer_bindings.shared_expert_down, device);
            cpu_shared_params.input_buffer_id = buffers.idFor(BufferId::NORMALIZED);
            cpu_shared_params.output_buffer_id = buffers.idFor(BufferId::MOE_SHARED_EXPERT_OUTPUT);
            cpu_shared_params.gate_scratch_buffer_id = buffers.idFor(BufferId::MOE_GATE_SCRATCH);
            cpu_shared_params.up_scratch_buffer_id = buffers.idFor(BufferId::MOE_UP_SCRATCH);
            shared_node = prefix + "shared_expert_ffn";
            graph.addNode(shared_node, ComputeStageFactory::createSharedExpertFFN(cpu_shared_params), device);
            graph.addDependency(shared_node, norm_node);
        }

        MoEExpertComputeStage::Params expert_params;
        expert_params.device_id = device;
        expert_params.input = buffers.normalized;
        expert_params.seq_len = rows;
        expert_params.d_model = config_.d_model;
        expert_params.num_experts = config_.moe.num_experts;
        expert_params.top_k = config_.moe.top_k;
        expert_params.routed_pipeline_kernel_owner = routed_kernel_owner;
        expert_params.cpu_router_q8_input_publication =
            CPURouterQ8InputPublicationPolicy::RequireRouterStage;
        expert_params.expert_intermediate = local_intermediate;
        expert_params.expert_weight_resolution_policy =
            MoEExpertWeightResolutionPolicy::PreparedRegistryOnly;
        expert_params.prepared_engine_binding_lifecycle =
            MoEPreparedEngineBindingLifecycle::CompleteAtConstruction;
        expert_params.persistent_weight_owner = weight_bindings_.physical_memory_owner;
        expert_params.local_expert_start = 0;
        expert_params.local_expert_count = -1;
        expert_params.layer_idx = layer_idx;
        expert_params.my_socket_id = std::max(0, config_.tp_device_idx);
        expert_params.participant_count = gtp->degree();
        expert_params.routed_assignment_policy = RoutedExpertAssignmentPolicy::StaticOwner;
        expert_params.routed_row_execution_policy = RoutedExpertRowExecutionPolicy::IntermediateShardLocal;
        expert_params.prepared_gate_gemm = std::move(prepared_gate);
        expert_params.prepared_up_gemm = std::move(prepared_up);
        expert_params.prepared_down_gemm = std::move(prepared_down);
        expert_params.moe_owned_kernels = std::move(owned_kernels);
        expert_params.expert_registry = &manager->expertGemmRegistry();
        expert_params.routing_indices = indices;
        expert_params.routing_weights = weights;
        expert_params.routing_indices_buffer_id = buffers.idFor(BufferId::MOE_EXPERT_INDICES);
        expert_params.routing_weights_buffer_id = buffers.idFor(BufferId::MOE_EXPERT_WEIGHTS);
        expert_params.force_decode_equivalent_verifier_prefill = force_verifier;
        expert_params.output = moe_output;
        expert_params.input_buffer_id = buffers.idFor(BufferId::NORMALIZED);
        expert_params.output_buffer_id = buffers.idFor(BufferId::MOE_COMBINED_OUTPUT);
        expert_params.prepared_store = prepared_weight_store_;
        expert_params.service_phase = phase;
        const std::string routed_node = prefix + "moe_expert_ffn";
        graph.addNode(routed_node, ComputeStageFactory::createMoEExpertCompute(expert_params), device);
        graph.addDependency(routed_node, router_node);

        const size_t allreduce_count =
            static_cast<size_t>(rows) * static_cast<size_t>(config_.d_model);
        const auto phase_override =
            predictor ? std::optional{ForwardExecutionPhase::Decode} : std::nullopt;
        std::string terminal;

        if (has_shared_expert && shared_requires_allreduce &&
            layer.shared_expert_gate_inp && buffers.attn_proj)
        {
            // Both routed and shared outputs are rank-local F/P partials while the
            // sigmoid gate vector and normalized hidden input are replicated across
            // every participant. Gate + combine locally first so one single [M, d_model]
            // AllReduce publishes the complete FFN delta (and M=1 decode lowers the
            // entire HC + router + shared + routed + gate region into RetainedCPUFFNStage).
            SharedExpertGateStage::Params gate_params;
            gate_params.device_id = device;
            gate_params.input = buffers.normalized;
            gate_params.gate_inp = layer.shared_expert_gate_inp;
            gate_params.shared_output = shared_output;
            gate_params.routed_residual = moe_output;
            gate_params.combined_output = buffers.attn_proj;
            gate_params.seq_len = rows;
            gate_params.d_model = config_.d_model;
            gate_params.execution_role = SharedExpertGateStage::ExecutionRole::ReplicatedOwner;
            gate_params.input_buffer_id = buffers.idFor(BufferId::NORMALIZED);
            gate_params.output_buffer_id = buffers.idFor(BufferId::MOE_SHARED_EXPERT_OUTPUT);
            gate_params.residual_buffer_id = buffers.idFor(BufferId::MOE_COMBINED_OUTPUT);
            gate_params.combined_output_buffer_id = buffers.idFor(BufferId::ATTN_PROJ);
            const std::string gate_node = prefix + "shared_expert_gate";
            graph.addNode(gate_node, ComputeStageFactory::createSharedExpertGate(gate_params), device);
            graph.addDependency(gate_node, shared_node);
            graph.addDependency(gate_node, routed_node);

            const std::string ar_node = prefix + "ffn_allreduce";
            auto ar_stage = createTPAllreduceStage(
                buffers.attn_proj, allreduce_count, device, layer_idx,
                /*is_attention=*/false, ar_node, buffers.idFor(BufferId::ATTN_PROJ),
                {}, std::nullopt, std::nullopt, phase_override);
            if (!ar_stage)
            {
                throw std::logic_error(
                    "CPU F-shard model FFN requires a non-null global TP AllReduce stage");
            }
            graph.addNode(ar_node, std::move(ar_stage), device);
            graph.addDependency(ar_node, gate_node);
            terminal = ar_node;
        }
        else
        {
            const std::string moe_ar_node = prefix + "moe_allreduce";
            auto moe_ar_stage = createTPAllreduceStage(
                moe_output, allreduce_count, device, layer_idx,
                /*is_attention=*/false, moe_ar_node, buffers.idFor(BufferId::MOE_COMBINED_OUTPUT),
                {}, std::nullopt, std::nullopt, phase_override);
            if (!moe_ar_stage)
            {
                throw std::logic_error(
                    "CPU F-shard model FFN requires a non-null global TP AllReduce stage");
            }
            graph.addNode(moe_ar_node, std::move(moe_ar_stage), device);
            graph.addDependency(moe_ar_node, routed_node);
            terminal = moe_ar_node;

            if (has_shared_expert)
            {
                std::string shared_last = shared_node;
                if (shared_requires_allreduce)
                {
                    const std::string shared_ar_node = prefix + "shared_expert_allreduce";
                    auto shared_ar_stage = createTPAllreduceStage(
                        shared_output, allreduce_count, device, layer_idx,
                        /*is_attention=*/false, shared_ar_node,
                        buffers.idFor(BufferId::MOE_SHARED_EXPERT_OUTPUT),
                        {}, std::nullopt, std::nullopt, phase_override);
                    if (!shared_ar_stage)
                    {
                        throw std::logic_error(
                            "CPU F-shard shared branch requires a non-null global TP AllReduce stage");
                    }
                    graph.addNode(shared_ar_node, std::move(shared_ar_stage), device);
                    graph.addDependency(shared_ar_node, shared_last);
                    graph.addDependency(shared_ar_node, terminal);
                    shared_last = shared_ar_node;
                }
                if (layer.shared_expert_gate_inp && buffers.attn_proj)
                {
                    SharedExpertGateStage::Params gate_params;
                    gate_params.device_id = device;
                    gate_params.input = buffers.normalized;
                    gate_params.gate_inp = layer.shared_expert_gate_inp;
                    gate_params.shared_output = shared_output;
                    gate_params.routed_residual = moe_output;
                    gate_params.combined_output = buffers.attn_proj;
                    gate_params.seq_len = rows;
                    gate_params.d_model = config_.d_model;
                    gate_params.execution_role = SharedExpertGateStage::ExecutionRole::ReplicatedOwner;
                    gate_params.input_buffer_id = buffers.idFor(BufferId::NORMALIZED);
                    gate_params.output_buffer_id = buffers.idFor(BufferId::MOE_SHARED_EXPERT_OUTPUT);
                    gate_params.residual_buffer_id = buffers.idFor(BufferId::MOE_COMBINED_OUTPUT);
                    gate_params.combined_output_buffer_id = buffers.idFor(BufferId::ATTN_PROJ);
                    const std::string gate_node = prefix + "shared_expert_gate";
                    graph.addNode(gate_node, ComputeStageFactory::createSharedExpertGate(gate_params), device);
                    graph.addDependency(gate_node, shared_last);
                    graph.addDependency(gate_node, terminal);
                    terminal = gate_node;
                }
                else
                {
                    ResidualAddStage::Params add_params;
                    add_params.device_id = device;
                    add_params.input = shared_output;
                    add_params.residual = moe_output;
                    add_params.output = buffers.attn_proj;
                    add_params.num_elements = allreduce_count;
                    add_params.input_buffer_id = buffers.idFor(BufferId::MOE_SHARED_EXPERT_OUTPUT);
                    add_params.residual_buffer_id = buffers.idFor(BufferId::MOE_COMBINED_OUTPUT);
                    add_params.output_buffer_id = buffers.idFor(BufferId::ATTN_PROJ);
                    const std::string combine_node = prefix + "moe_combine";
                    graph.addNode(combine_node, ComputeStageFactory::createResidualAdd(add_params), device);
                    graph.addDependency(combine_node, shared_last);
                    graph.addDependency(combine_node, terminal);
                    terminal = combine_node;
                }
            }
            else if (moe_output != buffers.attn_proj)
            {
                ResidualAddStage::Params copy_params;
                copy_params.device_id = device;
                copy_params.input = moe_output;
                copy_params.residual = nullptr;
                copy_params.output = buffers.attn_proj;
                copy_params.num_elements = allreduce_count;
                copy_params.input_buffer_id = buffers.idFor(BufferId::MOE_COMBINED_OUTPUT);
                copy_params.output_buffer_id = buffers.idFor(BufferId::ATTN_PROJ);
                const std::string combine_node = prefix + "moe_combine";
                graph.addNode(combine_node, ComputeStageFactory::createResidualAdd(copy_params), device);
                graph.addDependency(combine_node, terminal);
                terminal = combine_node;
            }
        }

        if (residual_read)
        {
            terminal = addBranchBlockWrite(
                graph, std::move(*residual_read), buffers, terminal,
                predictor ? &predictor->ffnOutput() : nullptr);
        }
        else if (layer_idx >= config_.pp_layer_offset + config_.n_layers - 1)
        {
            ResidualAddStage::Params res_params;
            res_params.device_id = device;
            res_params.input = buffers.attn_proj;
            res_params.residual = buffers.current_hidden;
            res_params.output = buffers.current_hidden;
            res_params.num_elements = allreduce_count;
            res_params.input_buffer_id = buffers.idFor(BufferId::ATTN_PROJ);
            res_params.residual_buffer_id = buffers.idFor(BufferId::HIDDEN_STATE);
            res_params.output_buffer_id = buffers.idFor(BufferId::HIDDEN_STATE);
            const std::string res_node = prefix + "ffn_residual";
            graph.addNode(res_node, ComputeStageFactory::createResidualAdd(res_params), device);
            graph.addDependency(res_node, terminal);
            terminal = res_node;
        }

        graph.setTerminalNode(terminal);
        return graph;
    }


    std::vector<Qwen35MoEGraph::MirroredLayerCheckpoint>
    Qwen35MoEGraph::mirroredLayerCheckpoints() const
    {
        std::vector<MirroredLayerCheckpoint> checkpoints;
        checkpoints.reserve(mirrored_layer_checkpoints_.size());
        for (const auto &[name, tensor] : mirrored_layer_checkpoints_)
        {
            if (!tensor)
                continue;
            checkpoints.push_back({
                .name = name,
                .tensor = tensor.get(),
            });
        }
        std::sort(
            checkpoints.begin(),
            checkpoints.end(),
            [](const MirroredLayerCheckpoint &lhs,
               const MirroredLayerCheckpoint &rhs)
            {
                return lhs.name < rhs.name;
            });
        return checkpoints;
    }

    std::string Qwen35MoEGraph::maybeAddEmbeddingDiagnosticCheckpoints(
        ComputeGraph &graph,
        TensorBase *source,
        BufferId source_buffer_id,
        const std::string &dependency,
        int total_tokens,
        DeviceId device)
    {
        if (mtpGraphContextActive() ||
            !device.is_gpu() ||
            !config_.grouped_mtp_verifier ||
            !mirroredLayerDiagnosticsEnabled())
        {
            return dependency;
        }
        if (!source || dependency.empty() || total_tokens <= 0)
        {
            throw std::runtime_error(
                "Qwen35 MoE grouped-verifier embedding diagnostics require "
                "an exact source, producer, and positive row count");
        }

        const std::string graph_shape =
            "grouped_verifier_m" + std::to_string(total_tokens);
        std::string checkpoint_dependency = dependency;
        for (int row = 0; row < total_tokens; ++row)
        {
            const std::string checkpoint_name =
                graph_shape + "_embedding_row" + std::to_string(row);
            auto &checkpoint =
                mirrored_layer_checkpoints_[checkpoint_name];
            if (!checkpoint)
            {
                checkpoint = std::make_unique<FP32Tensor>(
                    std::vector<size_t>{
                        1u,
                        static_cast<size_t>(config_.d_model)},
                    device);
                if (!checkpoint->allocateOnDevice(device))
                {
                    throw std::runtime_error(
                        "Qwen35 MoE could not allocate grouped-verifier "
                        "embedding checkpoint " +
                        checkpoint_name + " on " + device.to_string());
                }
            }

            const std::string node_name =
                "mirror_checkpoint_" + checkpoint_name;
            HiddenStateRowSelectStage::Params params;
            params.device_id = device;
            params.input = source;
            params.output = checkpoint.get();
            params.input_buffer_id = source_buffer_id;
            params.seq_len = total_tokens;
            params.d_model = config_.d_model;
            params.selected_row_idx = row;
            params.selection_policy =
                HiddenStateRowSelectStage::SelectionPolicy::FixedDeviceRow;

            graph.addNode(
                node_name,
                ComputeStageFactory::createHiddenStateRowSelect(params),
                device);
            graph.addDependency(node_name, checkpoint_dependency);
            checkpoint_dependency = node_name;
        }
        return checkpoint_dependency;
    }

    std::string Qwen35MoEGraph::maybeAddGDNDiagnosticCheckpoint(
        ComputeGraph &graph,
        const std::string &boundary,
        const ITensor *source,
        BufferId source_buffer_id,
        const std::string &dependency,
        int layer_idx,
        int total_tokens,
        int feature_dim,
        DeviceId device,
        const int32_t *sequence_lengths_device)
    {
        if (mtpGraphContextActive() ||
            !device.is_gpu() ||
            !mirroredLayerDiagnosticsEnabled())
        {
            return dependency;
        }
        if (!source || layer_idx < 0 ||
            total_tokens <= 0 || feature_dim <= 0)
        {
            throw std::runtime_error(
                "Qwen35 MoE GDN diagnostics require a source tensor and "
                "positive layer, row, and feature geometry");
        }

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
        const auto add_checkpoint =
            [&](const std::string &checkpoint_suffix,
                int selected_row,
                HiddenStateRowSelectStage::SelectionPolicy selection_policy,
                const int32_t *request_lengths,
                const std::string &input_dependency) -> std::string
        {
            const std::string checkpoint_name =
                graph_shape + "_layer" + std::to_string(layer_idx) +
                "_" + boundary + checkpoint_suffix;
            auto &checkpoint =
                mirrored_layer_checkpoints_[checkpoint_name];
            if (!checkpoint)
            {
                checkpoint = std::make_unique<FP32Tensor>(
                    std::vector<size_t>{
                        1u,
                        static_cast<size_t>(feature_dim)},
                    device);
                if (!checkpoint->allocateOnDevice(device))
                {
                    throw std::runtime_error(
                        "Qwen35 MoE could not allocate GDN diagnostic checkpoint " +
                        checkpoint_name + " on " + device.to_string());
                }
            }

            const std::string node_name =
                "layer" + std::to_string(layer_idx) +
                "_mirror_checkpoint_" + graph_shape + "_" + boundary +
                checkpoint_suffix;
            HiddenStateRowSelectStage::Params params;
            params.device_id = device;
            params.input = source;
            params.output = checkpoint.get();
            params.input_buffer_id = source_buffer_id;
            params.seq_len = total_tokens;
            params.d_model = feature_dim;
            params.selected_row_idx = selected_row;
            params.selection_policy = selection_policy;
            params.request_sequence_length_device = request_lengths;

            graph.addNode(
                node_name,
                ComputeStageFactory::createHiddenStateRowSelect(params),
                device);
            graph.addDependency(node_name, input_dependency);
            return node_name;
        };

        HiddenStateRowSelectStage::Params terminal_policy;
        terminal_policy.selected_row_idx = total_tokens - 1;
        configureMirroredCheckpointRowOwnership(
            terminal_policy,
            total_tokens,
            device,
            sequence_lengths_device);
        std::string checkpoint_dependency = add_checkpoint(
            "",
            terminal_policy.selected_row_idx,
            terminal_policy.selection_policy,
            terminal_policy.request_sequence_length_device,
            dependency);

        /*
         * Fixed rows are diagnostic observations of immutable graph geometry.
         * They deliberately do not consume mutable request-length metadata.
         * Keep the terminal checkpoint above device-owned so padded prefill
         * still identifies the exact logical final row.
         */
        const auto selected_layer = mirroredGDNDiagnosticLayer();
        if (selected_layer && *selected_layer == layer_idx &&
            graph_regime == "grouped_verifier")
        {
            /*
             * A grouped verifier graph is captured at its physical M, but the
             * final response budget may make only a strict prefix logical work.
             * Retain every physical row for the selected layer so diagnostics
             * can compare all valid rows and can also expose writes into padded
             * rows. The unsuffixed checkpoint above remains the authoritative
             * logical terminal row selected by resident request length.
             */
            for (int row = 0; row < total_tokens; ++row)
            {
                checkpoint_dependency = add_checkpoint(
                    "_row" + std::to_string(row),
                    row,
                    HiddenStateRowSelectStage::SelectionPolicy::FixedDeviceRow,
                    nullptr,
                    checkpoint_dependency);
            }
        }
        else if (graph_regime == "prefill" &&
                 selected_layer &&
                 *selected_layer == layer_idx)
        {
            static constexpr int kSampleRows[] = {
                0, 1, 2, 7, 31, 127, 511,
                575, 639, 703, 767, 831, 895, 959, 1023,
                2047};
            for (const int row : kSampleRows)
            {
                if (row >= total_tokens)
                    continue;
                checkpoint_dependency = add_checkpoint(
                    "_row" + std::to_string(row),
                    row,
                    HiddenStateRowSelectStage::SelectionPolicy::FixedDeviceRow,
                    nullptr,
                    checkpoint_dependency);
            }
        }
        return checkpoint_dependency;
    }

    std::string Qwen35MoEGraph::maybeAddFinalNormDiagnosticCheckpoint(
        ComputeGraph &graph,
        const std::string &boundary,
        TensorBase *source,
        BufferId source_buffer_id,
        const std::string &dependency,
        int total_tokens,
        DeviceId device,
        const int32_t *sequence_lengths_device)
    {
        if (mtpGraphContextActive() ||
            !device.is_gpu() ||
            !mirroredLayerDiagnosticsEnabled())
        {
            return dependency;
        }
        if (!source || total_tokens <= 0)
        {
            throw std::runtime_error(
                "Qwen35 MoE final-norm diagnostics require a source tensor "
                "and at least one logical row");
        }

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
        const std::string checkpoint_name =
            graph_shape + "_" + boundary;
        auto &checkpoint =
            mirrored_layer_checkpoints_[checkpoint_name];
        if (!checkpoint)
        {
            checkpoint = std::make_unique<FP32Tensor>(
                std::vector<size_t>{
                    1u,
                    static_cast<size_t>(config_.d_model)},
                device);
            /*
             * See the per-layer checkpoint allocation above. Final-norm
             * diagnostics are graph outputs as well and therefore require
             * prepared device storage before execution begins.
             */
            if (!checkpoint->allocateOnDevice(device))
            {
                throw std::runtime_error(
                    "Qwen35 MoE could not allocate final-norm diagnostic "
                    "checkpoint " +
                    checkpoint_name + " on " + device.to_string());
            }
        }

        const std::string node_name =
            "mirror_checkpoint_" + checkpoint_name;
        HiddenStateRowSelectStage::Params params;
        params.device_id = device;
        params.input = source;
        params.output = checkpoint.get();
        params.input_buffer_id = source_buffer_id;
        params.seq_len = total_tokens;
        params.d_model = config_.d_model;
        params.selected_row_idx = total_tokens - 1;
        configureMirroredCheckpointRowOwnership(
            params,
            total_tokens,
            device,
            sequence_lengths_device);

        graph.addNode(
            node_name,
            ComputeStageFactory::createHiddenStateRowSelect(params),
            device);
        if (!dependency.empty())
            graph.addDependency(node_name, dependency);
        return node_name;
    }

    HiddenStateRowSelectStage::SelectionPolicy
    Qwen35MoEGraph::mirroredCheckpointSelectionPolicy(
        int total_tokens,
        DeviceId device,
        const int32_t *sequence_lengths_device) const
    {
        /*
         * Both ordinary prefill and grouped verification may execute a strict
         * logical prefix of their captured physical M. Prefill does so through
         * bucket reuse; a grouped verifier does so when the remaining response
         * budget is smaller than the configured proposal depth. In either case,
         * the persistent device request length is the sole terminal-row owner.
         * Refusing to construct a multi-row GPU checkpoint without that owner
         * makes padded-row selection structurally impossible.
         */
        const bool resident_length_owned =
            device.is_gpu() &&
            total_tokens > 1 &&
            (config_.grouped_mtp_verifier ||
             !config_.live_mtp_request_batch_condition);
        if (!resident_length_owned)
        {
            return HiddenStateRowSelectStage::SelectionPolicy::
                FixedDeviceRow;
        }
        if (!sequence_lengths_device)
        {
            throw std::runtime_error(
                "Qwen35 MoE mirrored multi-row checkpoint requires a "
                "device-resident request length");
        }
        return HiddenStateRowSelectStage::SelectionPolicy::
            DeviceResidentRequestLength;
    }

    void Qwen35MoEGraph::configureMirroredCheckpointRowOwnership(
        HiddenStateRowSelectStage::Params &params,
        int total_tokens,
        DeviceId device,
        const int32_t *sequence_lengths_device) const
    {
        params.selection_policy =
            mirroredCheckpointSelectionPolicy(
                total_tokens,
                device,
                sequence_lengths_device);
        params.request_sequence_length_device =
            params.selection_policy ==
                    HiddenStateRowSelectStage::SelectionPolicy::
                        DeviceResidentRequestLength
                ? sequence_lengths_device
                : nullptr;
    }

} // namespace llaminar2
