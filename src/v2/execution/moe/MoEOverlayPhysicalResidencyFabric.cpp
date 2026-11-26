/**
 * @file MoEOverlayPhysicalResidencyFabric.cpp
 * @brief Physical residency fabric creation, initial slot adoption, and lane provisioning.
 */
#include "MoEOverlayPhysicalResidencyFabricInternal.h"

namespace llaminar2
{
    bool MoEOverlayProjectionWeightManifest::valid() const noexcept
    {
        const auto role = static_cast<std::size_t>(projection);
        return role < 3 && N > 0 && K > 0 && format.valid() &&
               (format.isFloating() || (K % 32) == 0);
    }

    bool MoEOverlayLayerWeightManifest::valid() const noexcept
    {
        if (layer_idx < 0)
            return false;
        std::array<bool, 3> seen{};
        for (const auto &projection : projections)
        {
            const auto role = static_cast<std::size_t>(projection.projection);
            if (!projection.valid() || role >= seen.size() || seen[role])
                return false;
            seen[role] = true;
        }
        return std::all_of(
            seen.begin(), seen.end(), [](bool value) { return value; });
    }

    bool MoEOverlayReusableContextSeal::valid() const noexcept
    {
        if (source_epoch == 0 || canonical_owner_map.owners().empty() ||
            canonical_owner_map.participants().empty() || local_banks.empty())
            return false;

        try
        {
            for (std::size_t bank_index = 0;
                 bank_index < local_banks.size();
                 ++bank_index)
            {
                const auto &bank = local_banks[bank_index];
                const auto *participant =
                    canonical_owner_map.participantForId(bank.participant_id);
                if (!participant || bank.epoch != source_epoch ||
                    bank.device != participant->device || bank.layers.empty() ||
                    bank.layers.front().resident_mask.empty() ||
                    !bank.valid(
                        bank.participant_id,
                        participant->device,
                        static_cast<int>(bank.layers.size()),
                        static_cast<int>(
                            bank.layers.front().resident_mask.size()),
                        bank.first_model_layer))
                {
                    return false;
                }
                canonical_owner_map.requireLayerGeometry(
                    static_cast<int>(bank.layers.size()),
                    static_cast<int>(bank.layers.front().resident_mask.size()),
                    bank.first_model_layer);
                for (std::size_t previous = 0;
                     previous < bank_index;
                     ++previous)
                {
                    if (local_banks[previous].participant_id ==
                        bank.participant_id)
                    {
                        return false;
                    }
                }

                for (std::size_t layer = 0;
                     layer < bank.layers.size();
                     ++layer)
                {
                    const auto &resident_mask =
                        bank.layers[layer].resident_mask;
                    for (std::size_t expert = 0;
                         expert < resident_mask.size();
                         ++expert)
                    {
                        const auto *owner = canonical_owner_map.placementFor(
                            bank.first_model_layer + static_cast<int>(layer), static_cast<int>(expert));
                        const auto target = owner ? MoEExpertPlacementTarget::fromPlacement(*owner) : std::nullopt;
                        bool member = false;
                        if (target)
                            for (size_t index = 0; index < target->participantCount(); ++index)
                                member = member || target->participantId(index) == bank.participant_id;
                        if (!target || resident_mask[expert] != member)
                        {
                            return false;
                        }
                    }
                }
            }
            return true;
        }
        catch (...)
        {
            /* Validation is a failure predicate at an noexcept lifecycle edge. */
            return false;
        }
    }

    /**
     * @brief Validate one native source contract through its immutable descriptor resolver.
     * @tparam Resolve Allocation-free descriptor resolver for the declared source authority.
     * @param resolve Returns the original native descriptor or null for an absent name.
     * @param num_layers Number of contiguous owned global layer identities.
     * @param num_experts Exact expert count required in every source projection.
     * @param first_model_layer First global source identity owned by this interval.
     * @return Validated native projection contracts without reading tensor payloads.
     * @throws std::invalid_argument For malformed intervals or incompatible metadata.
     */
    template<class Resolve>
    static std::vector<MoEOverlayLayerWeightManifest> buildExpertSourceManifest(
        const Resolve &resolve,
        int num_layers,
        int num_experts,
        int first_model_layer)
    {
        if (num_layers <= 0 || num_experts <= 0 || first_model_layer < 0 ||
            num_layers > std::numeric_limits<int>::max() - first_model_layer)
        {
            throw std::invalid_argument(
                "ExpertOverlay GGUF weight manifest requires positive model geometry and a valid owned layer interval");
        }

        const auto source_format = [](GGUFTensorType type)
            -> ExpertWeightFormat
        {
            const auto native = [](const NativeQuantisedFormatInfo &format)
            {
                return ExpertWeightFormat::nativeQuantised({
                    .codebook_id = format.codebook_id,
                    .is_superblock = format.is_superblock,
                    .present = true,
                });
            };
            switch (type)
            {
            case GGUFTensorType::Q4_0:
                return native(native_quantised_formats::Q4_0);
            case GGUFTensorType::Q4_1:
                return native(native_quantised_formats::Q4_1);
            case GGUFTensorType::Q5_0:
                return native(native_quantised_formats::Q5_0);
            case GGUFTensorType::Q5_1:
                return native(native_quantised_formats::Q5_1);
            case GGUFTensorType::Q8_0:
                return native(native_quantised_formats::Q8_0);
            case GGUFTensorType::Q2_K:
                return native(native_quantised_formats::Q2_K);
            case GGUFTensorType::Q3_K:
                return native(native_quantised_formats::Q3_K);
            case GGUFTensorType::Q4_K:
                return native(native_quantised_formats::Q4_K);
            case GGUFTensorType::Q5_K:
                return native(native_quantised_formats::Q5_K);
            case GGUFTensorType::Q6_K:
                return native(native_quantised_formats::Q6_K);
            case GGUFTensorType::Q8_K:
                return native(native_quantised_formats::Q8_K);
            case GGUFTensorType::IQ2_XXS:
                return native(native_quantised_formats::IQ2_XXS);
            case GGUFTensorType::IQ2_XS:
                return native(native_quantised_formats::IQ2_XS);
            case GGUFTensorType::IQ3_XXS:
                return native(native_quantised_formats::IQ3_XXS);
            case GGUFTensorType::IQ1_S:
                return native(native_quantised_formats::IQ1_S);
            case GGUFTensorType::IQ4_NL:
                return native(native_quantised_formats::IQ4_NL);
            case GGUFTensorType::IQ3_S:
                return native(native_quantised_formats::IQ3_S);
            case GGUFTensorType::IQ2_S:
                return native(native_quantised_formats::IQ2_S);
            case GGUFTensorType::IQ4_XS:
                return native(native_quantised_formats::IQ4_XS);
            case GGUFTensorType::IQ1_M:
                return native(native_quantised_formats::IQ1_M);
            case GGUFTensorType::F32:
                return ExpertWeightFormat::floating(TensorType::FP32);
            case GGUFTensorType::F16:
                return ExpertWeightFormat::floating(TensorType::FP16);
            case GGUFTensorType::BF16:
                return ExpertWeightFormat::floating(TensorType::BF16);
            }
            return {};
        };

        struct ProjectionTensor
        {
            ExpertTierWeightProjection projection;
            const char *suffix;
        };
        constexpr std::array<ProjectionTensor, 3> projections{{
            {ExpertTierWeightProjection::Gate, "ffn_gate_exps.weight"},
            {ExpertTierWeightProjection::Up, "ffn_up_exps.weight"},
            {ExpertTierWeightProjection::Down, "ffn_down_exps.weight"},
        }};

        std::vector<MoEOverlayLayerWeightManifest> result;
        result.reserve(static_cast<std::size_t>(num_layers));
        for (int layer_idx = first_model_layer;
             layer_idx < first_model_layer + num_layers; ++layer_idx)
        {
            MoEOverlayLayerWeightManifest layer;
            layer.layer_idx = layer_idx;
            for (std::size_t role = 0; role < projections.size(); ++role)
            {
                const std::string name =
                    "blk." + std::to_string(layer_idx) + "." +
                    projections[role].suffix;
                const GGUFTensorInfo *tensor = resolve(name);
                const ExpertWeightFormat format =
                    tensor ? source_format(tensor->type)
                           : ExpertWeightFormat{};
                if (!tensor || tensor->dimensions.size() != 3 ||
                    tensor->dimensions[0] == 0 ||
                    tensor->dimensions[1] == 0 ||
                    tensor->dimensions[2] !=
                        static_cast<std::uint64_t>(num_experts) ||
                    (!format.isFloating() &&
                     tensor->dimensions[0] % 32u != 0) ||
                    !format.valid() ||
                    tensor->dimensions[0] >
                        static_cast<std::uint64_t>(
                            std::numeric_limits<int>::max()) ||
                    tensor->dimensions[1] >
                        static_cast<std::uint64_t>(
                            std::numeric_limits<int>::max()))
                {
                    throw std::invalid_argument(
                        "ExpertOverlay GGUF tensor '" + name +
                        "' has absent, unsupported, or incompatible metadata");
                }

                /*
                 * ModelLoader canonicalizes GGUF dimensions to
                 * [K, N, experts]. Every destination engine consumes one
                 * expert as an ordinary [N, K] projection.
                 */
                layer.projections[role] = {
                    .projection = projections[role].projection,
                    .N = static_cast<int>(tensor->dimensions[1]),
                    .K = static_cast<int>(tensor->dimensions[0]),
                    .format = format,
                };
            }
            if (!layer.valid())
            {
                throw std::logic_error(
                    "ExpertOverlay GGUF metadata produced an invalid layer manifest");
            }
            result.push_back(std::move(layer));
        }
        return result;
    }

    std::vector<MoEOverlayLayerWeightManifest>
    buildMoEOverlayLayerWeightManifestFromGGUF(const GGUFModel &model,
        int num_layers, int num_experts, int first_model_layer)
    {
        return buildExpertSourceManifest([&model](const std::string &name) {
            return model.findTensor(name);
        }, num_layers, num_experts, first_model_layer);
    }

    std::vector<MoEOverlayLayerWeightManifest>
    buildMoEOverlayLayerWeightManifestFromLoader(const ModelLoader &loader,
        int num_layers, int num_experts, int first_model_layer)
    {
        return buildExpertSourceManifest([&loader](const std::string &name) {
            return loader.tensorInfo(name);
        }, num_layers, num_experts, first_model_layer);
    }


    std::shared_ptr<MoEOverlayPhysicalResidencyFabric>
    MoEOverlayPhysicalResidencyFabric::create(Config config)
    {
        if (!config.memory_authority || !config.registry ||
            !config.initial_snapshot ||
            !config.initial_snapshot->valid())
        {
            throw std::invalid_argument(
                "ExpertOverlay physical fabric requires the rank-bound memory authority, registry, and valid initial snapshot");
        }
        if (config.initial_snapshot->ownershipKind() == MoEOverlayResidencyOwnershipKind::JointFragments &&
            config.inventory_authority != MoEOverlayPhysicalInventoryAuthority::ParticipantRegistry &&
            std::any_of(config.initial_snapshot->placementOwnerMap().participants().begin(),
                config.initial_snapshot->placementOwnerMap().participants().end(),
                [](const auto &participant) { return participant.device.is_cpu(); }))
            throw std::invalid_argument("CPU-involved cohort fabric requires its original participant-registry physical authority");
        if (config.registry->initialEpoch() !=
                config.initial_snapshot->epoch ||
            !config.registry->allInitialBanksReady())
        {
            throw std::invalid_argument(
                "ExpertOverlay physical fabric requires complete matching initial banks");
        }
        for(const auto id:config.registry->localParticipantIds()) {
            const auto endpoint=config.registry->endpoint(id);
            if(endpoint && endpoint->device().is_gpu() &&
                (!config.physical_gpu_lane_demand || config.physical_gpu_lane_demand->original_rank!=config.memory_authority->worldRank()))
                throw std::invalid_argument("Physical GPU fabric requires its original rank-bound admitted lane declaration");
        }
        if (config.shadow_slots_per_endpoint_layer == 0 ||
            config.staging_capacity_bytes == 0 ||
            config.maximum_concurrent_cycles == 0 ||
            config.maximum_cycles_per_layer == 0 ||
            config.maximum_execution_streams == 0 ||
            config.maximum_execution_streams >
                config.maximum_concurrent_cycles)
        {
            throw std::invalid_argument(
                "ExpertOverlay physical fabric requires positive shadow, staging, cycle, and execution-stream capacities, with streams no greater than cycles");
        }
        if (config.remote_projection_transport)
        {
            if (config.staging_capacity_bytes >
                static_cast<std::size_t>(
                    std::numeric_limits<std::uint32_t>::max()))
            {
                throw std::invalid_argument(
                    "ExpertOverlay remote staging capacity exceeds the canonical 32-bit chunk ABI");
            }
            const int local_world_rank =
                config.remote_projection_transport->worldRank();
            for (const int participant_id :
                 config.registry->localParticipantIds())
            {
                const auto *participant =
                    config.initial_snapshot->placementOwnerMap().participantForId(
                        participant_id);
                if (!participant || !participant->world_rank_known ||
                    participant->world_rank != local_world_rank)
                {
                    throw std::invalid_argument(
                        "ExpertOverlay distributed physical fabric registry contains a participant assigned to another or unknown rank");
                }
            }
        }
        if (config.perf_device.empty())
            config.perf_device = "expert_overlay";

        auto impl = std::make_unique<Impl>();
        materializeEndpointPools(config, *impl);
        materializeTransferLanes(config, *impl);

        PerfStatsCollector::addCounter(
            "moe_overlay_residency",
            "physical_fabrics_materialized",
            1.0,
            "model_setup",
            config.perf_device,
            {{"endpoint_layer_pools",
              std::to_string(impl->endpoint_pools.size())},
             {"endpoint_geometry_pools",
              std::to_string(
                  impl->endpoint_geometry_pools.load(
                      std::memory_order_relaxed))},
             {"adopted_initial_slots",
              std::to_string(
                  impl->adopted_initial_slots.load(
                      std::memory_order_relaxed))},
             {"persistent_lanes",
              std::to_string(
                  impl->persistent_transfer_lanes.load(
                      std::memory_order_relaxed))},
             {"maximum_parallel_edge_lanes",
              std::to_string(
                  impl->maximum_parallel_edge_lanes.load(
                      std::memory_order_relaxed))},
             {"maximum_parallel_cpu_copy_lanes",
              std::to_string(
                  impl->maximum_parallel_cpu_copy_lanes.load(
                      std::memory_order_relaxed))},
             {"maximum_concurrent_cycles",
              std::to_string(config.maximum_concurrent_cycles)},
             {"maximum_execution_streams",
              std::to_string(config.maximum_execution_streams)}});
        return std::shared_ptr<MoEOverlayPhysicalResidencyFabric>(
            new MoEOverlayPhysicalResidencyFabric(
                std::move(config), std::move(impl)));
    }

    MoEOverlayPhysicalResidencyFabric::MoEOverlayPhysicalResidencyFabric(
        Config config,
        std::unique_ptr<Impl> impl) noexcept
        : config_(std::move(config)), impl_(std::move(impl))
    {
    }

    MoEOverlayPhysicalResidencyFabric::~MoEOverlayPhysicalResidencyFabric() =
        default;

    MoEOverlayParticipantPreparedTransfers
    MoEOverlayPhysicalResidencyFabric::prepareTransfers(
        const MoEOverlayResidencyTransaction &transaction,
        const std::vector<int> &local_destination_participants)
    {
        if (!transaction.valid() || transaction.empty() ||
            !transaction.candidate)
        {
            impl_->waves_failed.fetch_add(1, std::memory_order_relaxed);
            return {
                .status = MoEOverlayResidencyStageStartStatus::Failed,
                .error =
                    "ExpertOverlay physical fabric requires a valid non-empty transaction",
            };
        }
        return preparePhysicalTransfers(
            transaction.purpose,
            transaction.expected_epoch,
            transaction.candidate->epoch,
            fingerprintMoEOverlayResidencyExecutionPlan(transaction),
            transaction.migrations,
            transaction.migration_cycles,
            transaction.shadow_requirements,
            transaction.candidate.get(),
            nullptr,
            &transaction,
            local_destination_participants);
    }

    MoEOverlayParticipantPreparedTransfers
    MoEOverlayPhysicalResidencyFabric::prepareDeviceTransfers(
        const MoEOverlayDevicePhysicalMovementBatch &batch,
        const std::vector<int> &local_destination_participants)
    {
        if (!batch.valid() || !batch.movesWeights() ||
            (batch.kind !=
                 MoEOverlayDeviceControllerTransactionKind::DynamicPlacement &&
             batch.kind != MoEOverlayDeviceControllerTransactionKind::
                               PreparedContextRestore))
        {
            impl_->waves_failed.fetch_add(1, std::memory_order_relaxed);
            return {
                .status = MoEOverlayResidencyStageStartStatus::Failed,
                .error =
                    "ExpertOverlay physical fabric requires non-empty device-authored durable movement",
            };
        }
        if (!impl_->device_slot_ledger)
        {
            impl_->waves_failed.fetch_add(1, std::memory_order_relaxed);
            return {
                .status = MoEOverlayResidencyStageStartStatus::Failed,
                .error =
                    "ExpertOverlay physical fabric lost its device slot ledger",
            };
        }
        std::string ledger_error;
        if (!impl_->device_slot_ledger->begin(batch, &ledger_error))
        {
            impl_->waves_failed.fetch_add(1, std::memory_order_relaxed);
            return {
                .status = MoEOverlayResidencyStageStartStatus::Failed,
                .error = std::move(ledger_error),
            };
        }

        auto prepared = preparePhysicalTransfers(
            MoEOverlayResidencyTransactionPurpose::PlacementChange,
            batch.base_epoch,
            batch.candidate_epoch,
            batch.execution_fingerprint,
            batch.migrations,
            batch.migration_cycles,
            batch.shadow_requirements,
            nullptr,
            &batch,
            nullptr,
            local_destination_participants);
        if (prepared.status != MoEOverlayResidencyStageStartStatus::Started)
        {
            std::string abort_error;
            if (!impl_->device_slot_ledger->abort(batch, &abort_error) &&
                prepared.error.empty())
            {
                prepared.error = std::move(abort_error);
                prepared.status = MoEOverlayResidencyStageStartStatus::Failed;
            }
        }
        return prepared;
    }

    MoEOverlayParticipantPreparedTransfers
    MoEOverlayPhysicalResidencyFabric::preparePhysicalTransfers(
        MoEOverlayResidencyTransactionPurpose purpose,
        std::uint64_t expected_epoch,
        std::uint64_t candidate_epoch,
        const MoEOverlayResidencyExecutionFingerprint &execution_fingerprint,
        const std::vector<MoEOverlayTierMigration> &migrations,
        const std::vector<MoEOverlayTierMigrationCycle> &migration_cycles,
        const std::vector<MoEOverlayTierShadowRequirement> &shadow_requirements,
        const MoEOverlayResidencySnapshot *candidate,
        const MoEOverlayDevicePhysicalMovementBatch *device_batch,
        const MoEOverlayResidencyTransaction *original_transaction,
        const std::vector<int> &local_destination_participants)
    {
        MoEOverlayParticipantPreparedTransfers result;
        auto fail = [&](std::string message)
            -> MoEOverlayParticipantPreparedTransfers
        {
            impl_->waves_failed.fetch_add(1, std::memory_order_relaxed);
            result.status = MoEOverlayResidencyStageStartStatus::Failed;
            result.error = std::move(message);
            return std::move(result);
        };

        {
            std::lock_guard<std::mutex> seal_lock(
                impl_->reusable_seal_mutex);
            if (impl_->reusable_seal_state !=
                Impl::ReusableSealState::Open)
            {
                return fail(
                    "ExpertOverlay physical movement is forbidden after reusable-context sealing begins");
            }
        }

        if (migrations.empty() || expected_epoch == 0u ||
            candidate_epoch == 0u || !execution_fingerprint.valid())
            return fail(
                "ExpertOverlay physical fabric received incomplete physical movement identity");

        const auto configured_local = config_.registry->localParticipantIds();
        if (configured_local != local_destination_participants)
            return fail(
                "ExpertOverlay physical fabric received a different local participant set");

        const auto is_local_participant = [&](int participant_id)
        {
            return std::binary_search(
                configured_local.begin(),
                configured_local.end(),
                participant_id);
        };
        const auto remote_transport = config_.remote_projection_transport;
        const int world_rank = remote_transport
                                   ? remote_transport->worldRank()
                                   : -1;
        const int world_size = remote_transport
                                   ? remote_transport->worldSize()
                                   : 0;
        bool has_remote_migrations = false;

        // Normalize each logical edge once. Every subsequent physical demand
        // and operation uses this original role/member order, never the absent
        // scalar owner of a cohort or allocation-capacity sized work.
        std::vector<std::vector<MoEOverlayPhysicalProjectionPiece>> physical_pieces;
        std::vector<std::shared_ptr<const MoEOverlayMigrationMeasurementGeometry>> measurement_geometries;
        MoEOverlayPhysicalRemotePieceOrdinals remote_piece_ordinals;
        try
        {
            physical_pieces.reserve(migrations.size());
            measurement_geometries.reserve(migrations.size());
            for (const auto &migration : migrations)
            {
                const auto source = std::find_if(impl_->original_layer_manifests.begin(), impl_->original_layer_manifests.end(),
                    [&](const auto &entry) { return entry.layer_idx == migration.layer_idx; });
                if (source == impl_->original_layer_manifests.end())
                    return fail("Physical movement lost its original authenticated layer source");
                const auto geometry = makePhysicalMoEOverlayMeasurementGeometry(migration, *source,
                    originalFabricGeometry(config_).expertCount(), config_.initial_snapshot->placementOwnerMap());
                physical_pieces.push_back(physicalMoEOverlayProjectionPieces(geometry->targets().first, geometry->targets().second));
                measurement_geometries.push_back(geometry);
            }
            remote_piece_ordinals = physicalMoEOverlayRemotePieceOrdinals(physical_pieces,
                config_.initial_snapshot->placementOwnerMap(), remote_transport ? world_size : 1);
        }
        catch (const std::exception &error)
        {
            return fail(error.what());
        }
        // Cohort calibration borrows the genuine shadow-only transaction. Its
        // live map is unchanged, so it must not enter the candidate-placement
        // arrival producer or manufacture a placement change to pass that guard.
        const bool has_cohort = std::any_of(migrations.begin(), migrations.end(),
            [](const auto &edge) { return edge.source.shard_cohort || edge.destination.shard_cohort; });
        if (has_cohort && purpose == MoEOverlayResidencyTransactionPurpose::EconomyCalibration)
        {
            if (device_batch || !original_transaction || !original_transaction->valid() ||
                original_transaction->purpose != purpose || original_transaction->expected_epoch != expected_epoch ||
                original_transaction->candidate.get() != candidate || candidate->epoch != candidate_epoch ||
                &original_transaction->migrations != &migrations)
                return fail("CPU cohort calibration requires its exact original unpublishable shadow transaction");
        }
        else if (has_cohort && purpose != MoEOverlayResidencyTransactionPurpose::PlacementChange &&
                 purpose != MoEOverlayResidencyTransactionPurpose::PreparedContextRestoration)
            return fail("Cohort physical movement has an unsupported transaction purpose");

        try
        {
            requireSimpleMoEOverlayPhysicalCycles(migrations, migration_cycles);
            const auto lane_scope=remote_transport
                ?MoEOverlayPhysicalLaneScope::communicator(world_size)
                :MoEOverlayPhysicalLaneScope::rankLocal(config_.memory_authority->worldRank());
            MoEOverlayPhysicalLaneCounts actual_demand;
            for (size_t index = 0; index < migrations.size(); ++index)
            {
                const auto source = std::find_if(impl_->original_layer_manifests.begin(), impl_->original_layer_manifests.end(),
                    [&](const auto &entry) { return entry.layer_idx == migrations[index].layer_idx; });
                if (source == impl_->original_layer_manifests.end())
                    throw std::logic_error("Physical movement lost its original layer manifest");
                const auto demand = physicalMoEOverlayLaneCounts(physical_pieces[index], migrations[index], *source,
                    originalFabricGeometry(config_).expertCount(), config_.initial_snapshot->placementOwnerMap(),
                    lane_scope);
                for (const auto &[key, count] : demand)
                    actual_demand[key] = checkedMoEOverlayLaneAdd(actual_demand[key], count);
            }
            requireFitsMoEOverlayLaneDemand(impl_->lane_demand_plan, actual_demand);
        }
        catch (const std::exception &error)
        {
            return fail(error.what());
        }
        for (const auto &pieces : physical_pieces)
            for (const auto &piece : pieces)
            {
                const auto &source = *config_.initial_snapshot->placementOwnerMap().participantForId(piece.source_participant);
                const auto &destination = *config_.initial_snapshot->placementOwnerMap().participantForId(piece.destination_participant);
                const bool source_local = is_local_participant(source.participant_id);
                const bool destination_local = is_local_participant(destination.participant_id);
                if (!remote_transport)
                {
                    if (!source_local || !destination_local)
                        return fail("ExpertOverlay transaction contains a remote endpoint but no remote projection transport is installed");
                    continue;
                }
                if (!source.world_rank_known || !destination.world_rank_known ||
                    source.world_rank < 0 || source.world_rank >= world_size ||
                    destination.world_rank < 0 || destination.world_rank >= world_size)
                    return fail("ExpertOverlay distributed movement requires exact in-range world-rank ownership");
                if (source_local != (source.world_rank == world_rank) ||
                    destination_local != (destination.world_rank == world_rank))
                    return fail("ExpertOverlay local participant registry disagrees with distributed rank ownership");
                has_remote_migrations = has_remote_migrations || source.world_rank != destination.world_rank;
            }

        if (migration_cycles.size() >
            config_.maximum_concurrent_cycles)
        {
            return fail(
                "ExpertOverlay transaction exceeds the physical fabric's concurrent-cycle lane BOM");
        }

        /*
         * Prove the complete software-dispatch fan-out before reserving a
         * destination slot. A Started wave is therefore guaranteed to own one
         * distinct physical lane for every non-CPU projection; no operation is
         * admitted on the promise that an earlier operation will finish first.
         */
        std::map<EdgeKey, std::size_t> local_lane_demand;
        std::map<CpuAddressEdgeKey, std::size_t> cpu_copy_lane_demand;
        std::map<RemoteGpuLaneKey, std::size_t> remote_gpu_lane_demand;
        for (const auto &pieces : physical_pieces)
            for (const auto &piece : pieces)
            {
                const auto &source = *config_.initial_snapshot->placementOwnerMap().participantForId(piece.source_participant);
                const auto &destination = *config_.initial_snapshot->placementOwnerMap().participantForId(piece.destination_participant);
                const bool source_local = is_local_participant(source.participant_id);
                const bool destination_local = is_local_participant(destination.participant_id);
                const auto projection = piece.projection;
                if (source_local && destination_local && source.device.is_cpu() && destination.device.is_cpu())
                    ++cpu_copy_lane_demand[CpuAddressEdgeKey{
                        .source = source.address, .destination = destination.address, .projection = projection}];
                else if (source_local && destination_local)
                    ++local_lane_demand[EdgeKey{
                        .source = source.device, .destination = destination.device, .projection = projection}];
                if (has_remote_migrations && source_local && !destination_local && source.device.is_gpu())
                    ++remote_gpu_lane_demand[RemoteGpuLaneKey{
                        .device = source.device, .projection = projection, .role = RemoteGpuLaneRole::Source}];
                if (has_remote_migrations && !source_local && destination_local && destination.device.is_gpu())
                    ++remote_gpu_lane_demand[RemoteGpuLaneKey{
                        .device = destination.device, .projection = projection, .role = RemoteGpuLaneRole::Destination}];
            }
        for (const auto &[edge, demand] : local_lane_demand)
        {
            std::size_t capacity = 0;
            if (const auto peer_pool = impl_->peer_lanes.find(edge);
                peer_pool != impl_->peer_lanes.end())
            {
                capacity = peer_pool->second->capacity();
            }
            else if (const auto blob_pool = impl_->blob_lanes.find(edge);
                     blob_pool != impl_->blob_lanes.end())
            {
                capacity = blob_pool->second->capacity();
            }
            else
            {
                const auto weight_pool = impl_->weight_lanes.find(edge);
                if (weight_pool != impl_->weight_lanes.end())
                    capacity = weight_pool->second->capacity();
            }
            if (demand > capacity)
            {
                return fail(
                    "ExpertOverlay transaction exceeds an admission-sized local parallel lane pool");
            }
        }
        std::size_t local_cpu_copy_operation_count = 0;
        for (const auto &[edge, demand] : cpu_copy_lane_demand)
        {
            const auto pool = impl_->cpu_copy_lanes.find(edge);
            const std::size_t capacity =
                pool == impl_->cpu_copy_lanes.end()
                    ? 0
                    : pool->second->capacity();
            if (demand > capacity)
            {
                return fail(
                    "ExpertOverlay transaction exceeds an admission-sized CPU parallel worker pool");
            }
            if (demand >
                std::numeric_limits<std::size_t>::max() -
                    local_cpu_copy_operation_count)
            {
                return fail(
                    "ExpertOverlay CPU parallel worker demand overflows size_t");
            }
            local_cpu_copy_operation_count += demand;
        }
        for (const auto &[key, demand] : remote_gpu_lane_demand)
        {
            const auto pool = impl_->remote_gpu_lanes.find(key);
            const std::size_t capacity = pool == impl_->remote_gpu_lanes.end()
                ? 0
                : pool->second.size();
            if (demand > capacity)
            {
                return fail(
                    "ExpertOverlay transaction exceeds an admission-sized remote GPU parallel lane pool");
            }
        }

        /*
         * Validate the immutable BOM and transient availability for the whole
         * cycle set before acquiring one slot.  Exceeding the model-time BOM is
         * fatal; an old ticket retaining a planned slot is ordinary deferral.
         */
        struct GeometryShadowDemand
        {
            EndpointLayerPool *binding = nullptr;
            std::size_t slot_count = 0;
            int destination_participant = -1;
        };
        std::map<const EndpointGeometryPool *, GeometryShadowDemand>
            demand_by_geometry;
        for (const auto &requirement : shadow_requirements)
        {
            if (!std::binary_search(
                    configured_local.begin(),
                    configured_local.end(),
                    requirement.destination_participant))
            {
                continue;
            }
            EndpointLayerPool *pool = nullptr;
            try
            {
                pool = &requireEndpointPool(
                    *impl_,
                    requirement.destination_participant,
                    requirement.layer_idx);
            }
            catch (const std::exception &error)
            {
                return fail(error.what());
            }
            if (!pool->geometry_pool ||
                requirement.slot_count > pool->layer_arrival_capacity)
            {
                return fail(
                    "ExpertOverlay migration wave exceeds the per-layer shadow arrival bound: participant=" +
                    std::to_string(requirement.destination_participant) +
                    " layer=" + std::to_string(requirement.layer_idx) +
                    " required=" + std::to_string(requirement.slot_count) +
                    " capacity=" +
                    std::to_string(pool->layer_arrival_capacity));
            }
            auto &geometry_demand = demand_by_geometry[
                pool->geometry_pool.get()];
            if (!geometry_demand.binding)
            {
                geometry_demand.binding = pool;
                geometry_demand.destination_participant =
                    requirement.destination_participant;
            }
            if (requirement.slot_count >
                std::numeric_limits<std::size_t>::max() -
                    geometry_demand.slot_count)
            {
                return fail(
                    "ExpertOverlay geometry shadow demand overflows size_t");
            }
            geometry_demand.slot_count += requirement.slot_count;
        }

        for (const auto &[_, requirement] : demand_by_geometry)
        {
            if (!requirement.binding ||
                !requirement.binding->geometry_pool)
            {
                return fail(
                    "ExpertOverlay geometry shadow demand lost its typed arena binding");
            }
            const auto &geometry = *requirement.binding->geometry_pool;
            if (requirement.slot_count > geometry.shadow_capacity)
            {
                return fail(
                    "ExpertOverlay migration wave exceeds the shared exact-geometry shadow-slot BOM: participant=" +
                    std::to_string(requirement.destination_participant) +
                    " required=" + std::to_string(requirement.slot_count) +
                    " capacity=" +
                    std::to_string(geometry.shadow_capacity));
            }
            const std::size_t available = availableSlots(
                *requirement.binding);
            if (requirement.slot_count > available)
            {
                impl_->waves_deferred.fetch_add(
                    1, std::memory_order_relaxed);
                result.status = MoEOverlayResidencyStageStartStatus::Deferred;
                result.error =
                    "ExpertOverlay shadow slots are retained by an older epoch: "
                    "participant=" +
                    std::to_string(requirement.destination_participant) +
                    " required=" + std::to_string(requirement.slot_count) +
                    " available=" + std::to_string(available) +
                    " planned_capacity=" +
                    std::to_string(geometry.shadow_capacity) +
                    " expected_epoch=" +
                    std::to_string(expected_epoch) +
                    " candidate_epoch=" +
                    std::to_string(candidate_epoch);
                return result;
            }
        }

        const ExpertTierDeviceTimingCapture wave_timing_capture =
            (config_.collect_economy_measurements &&
             (config_.economy_measurement_scope ==
                  MoEOverlayMigrationMeasurementScope::AllWaves ||
              purpose ==
                  MoEOverlayResidencyTransactionPurpose::EconomyCalibration))
                ? ExpertTierDeviceTimingCapture::Enabled
                : ExpertTierDeviceTimingCapture::Disabled;

        std::vector<std::map<int, ReservedDestination>> destinations(migrations.size());
        std::vector<std::map<int, std::shared_ptr<MoEOverlayPreparedExpertArrival>>> destination_arrivals(migrations.size());
        std::vector<std::map<int, std::shared_ptr<RetainedPhysicalSourceMember>>> source_members(migrations.size());
        try
        {
            for (std::size_t migration_index = 0;
                 migration_index < migrations.size();
                 ++migration_index)
            {
                const auto &migration =
                    migrations[migration_index];
                const auto target = MoEExpertPlacementTarget::fromPlacement(migration.destination);
                const auto origin = MoEExpertPlacementTarget::fromPlacement(migration.source);
                if (!target || !origin) throw std::invalid_argument("Physical wave lost its original placement alternatives");
                for (size_t member = 0; member < target->participantCount(); ++member)
                {
                    const int participant = target->participantId(member);
                    if (!is_local_participant(participant)) continue;
                    auto &pool = requireEndpointPool(*impl_, participant, migration.layer_idx);
                    destinations[migration_index].emplace(participant,
                        reserveDestination(pool, migration.layer_idx, migration.expert_id, candidate_epoch));
                    std::shared_ptr<MoEOverlayPreparedExpertArrival> arrival;
                    if (migration.destination.shard_cohort)
                    {
                        std::string arrival_error;
                        if (device_batch)
                            arrival = config_.registry->prepareJointDeviceArrival(*device_batch,
                                migration_index, participant, &arrival_error);
                        else if (!candidate || candidate->epoch != candidate_epoch)
                            throw std::invalid_argument("Fragment reservation requires its actual original candidate epoch");
                        else if (purpose == MoEOverlayResidencyTransactionPurpose::EconomyCalibration)
                            arrival = config_.registry->prepareJointCalibrationArrival(*original_transaction,
                                migration_index, participant, &arrival_error);
                        else
                            arrival = config_.registry->prepareJointArrival(*candidate, participant,
                                migration.layer_idx, migration.expert_id, &arrival_error);
                        if (!arrival) throw std::invalid_argument(arrival_error);
                    }
                    else arrival = std::make_shared<MoEOverlayPreparedExpertArrival>();
                    destination_arrivals[migration_index].emplace(participant, std::move(arrival));
                }
                if (!device_batch)
                    for (size_t member = 0; member < origin->participantCount(); ++member)
                    {
                        const int participant = origin->participantId(member);
                        if (!is_local_participant(participant)) continue;
                        auto source = std::make_shared<RetainedPhysicalSourceMember>();
                        source->endpoint = config_.registry->endpoint(participant);
                        source->bank = source->endpoint ? source->endpoint->acquire(expected_epoch)
                                                       : MoEOverlayParticipantBankLease{};
                        if (!source->endpoint || !source->bank ||
                            !source->bank->containsModelLayer(migration.layer_idx) || migration.expert_id < 0 ||
                            migration.expert_id >= source->endpoint->numExperts())
                            throw std::invalid_argument("Physical wave cannot retain its exact original source member epoch");
                        const auto &layer = source->bank->layerForModelLayer(migration.layer_idx);
                        if (bool(layer.fragment_family) != bool(migration.source.shard_cohort))
                            throw std::invalid_argument("Physical source bank has a different whole/cohort ownership arm");
                        if (layer.fragment_family)
                        {
                            const auto &actual = layer.fragment_family->cohort();
                            const auto &expected = migration.source.shard_cohort;
                            const auto member = layer.fragment_family->participantIndex();
                            if (!actual || actual.get() != expected.get() || actual.owner_before(expected) ||
                                expected.owner_before(actual) || member >= actual->participants().size() ||
                                actual->participants()[member].participant_id != participant)
                                throw std::invalid_argument("Physical source bank lost its SAME original cohort/member identity");
                        }
                        if (!layer.resident_mask.at(size_t(migration.expert_id)))
                            throw std::invalid_argument("Physical source expert is absent from its original epoch bank");
                        for (const auto role : kProjections)
                            source->engines[projectionIndex(role)] = projectionEngine(layer, size_t(migration.expert_id), role);
                        const auto count = std::count_if(physical_pieces[migration_index].begin(), physical_pieces[migration_index].end(),
                            [participant](const auto &piece) { return piece.source_participant == participant; });
                        if (count <= 0) throw std::invalid_argument("Physical source member has no actual projection work");
                        source->pending.store(size_t(count));
                        source_members[migration_index].emplace(participant, std::move(source));
                    }
            }
        }
        catch (const std::exception &error)
        {
            return fail(error.what());
        }

        std::vector<std::map<int, std::array<DestinationProjectionGroup, 3>>> destination_groups(migrations.size());
        try
        {
        for (size_t index = 0; index < migrations.size(); ++index)
            for (const auto &[participant, arrival] : destination_arrivals[index])
                for (const auto role : kProjections)
                {
                    auto &group = destination_groups[index][participant][projectionIndex(role)];
                    group.storage = std::make_shared<DestinationProjectionStorage>();
                    auto &reserved = destinations[index].at(participant);
                    if (reserved.cpu) group.storage->engine = cpuProjection(*reserved.cpu, role).engine;
                    std::vector<ExpertTierWeightProjectionIntersection> coverage;
                    for (const auto &piece : physical_pieces[index])
                        if (piece.destination_participant == participant && piece.projection == role)
                            coverage.push_back(piece.intersection);
                    group.completion = std::make_shared<MoEOverlayPreparedProjectionCompletion>(std::move(coverage),
                        arrival, role, [storage = group.storage] { return storage->engine; });
                }

        }
        catch (const std::exception &error)
        {
            return fail(error.what());
        }

        result.migrations.resize(migrations.size());
        struct RemoteOperationTarget
        {
            std::size_t migration_index = 0;
            std::size_t operation_index = 0;
            std::size_t piece_index = 0;
            std::shared_ptr<MoEOverlayPreparedProjectionCompletion> completion;
        };
        std::vector<MoEOverlayMPIRemoteProjectionBinding> remote_bindings;
        std::vector<RemoteOperationTarget> remote_targets;
        if (has_remote_migrations)
        {
            size_t remote_projection_count = 0;
            for (const auto &pieces : physical_pieces)
                remote_projection_count = checkedMoEOverlayLaneAdd(remote_projection_count, pieces.size());
            remote_bindings.reserve(remote_projection_count);
            remote_targets.reserve(remote_projection_count);
        }

        const auto remote_execution_fingerprint = remote_transport
            ? std::optional<MoEOverlayResidencyExecutionFingerprint>(
                  execution_fingerprint)
            : std::nullopt;
        const auto cpu_copy_wave_gate = local_cpu_copy_operation_count == 0
            ? std::shared_ptr<CpuCopyWaveGate>{}
            : std::make_shared<CpuCopyWaveGate>(
                  local_cpu_copy_operation_count);
        std::uint64_t local_cpu_copy_operations = 0;
        std::uint64_t local_remote_cpu_operations = 0;
        std::uint64_t local_remote_gpu_cpu_operations = 0;
        std::uint64_t local_remote_gpu_blob_operations = 0;
        std::uint64_t local_gpu_cpu_operations = 0;
        std::uint64_t local_same_backend_gpu_operations = 0;
        std::uint64_t local_heterogeneous_gpu_operations = 0;
        std::map<RemoteGpuLaneKey, std::size_t> remote_gpu_lane_cursor;
        try
        {
            for (std::size_t migration_index = 0;
                 migration_index < migrations.size();
                 ++migration_index)
            {
                const auto &migration =
                    migrations[migration_index];
                auto &prepared = result.migrations[migration_index];
                const auto &pieces = physical_pieces[migration_index];
                prepared.operations.resize(pieces.size());
                prepared.measurement_geometry = std::move(measurement_geometries[migration_index]);
                const auto destination_target = *MoEExpertPlacementTarget::fromPlacement(migration.destination);
                for (size_t member = 0; member < destination_target.participantCount(); ++member)
                {
                    const int participant = destination_target.participantId(member);
                    const auto arrival = destination_arrivals[migration_index].find(participant);
                    if (arrival != destination_arrivals[migration_index].end())
                        prepared.destination_arrivals.push_back({participant, arrival->second});
                }
                for (size_t operation_index = 0; operation_index < pieces.size(); ++operation_index)
                {
                    const auto &piece = pieces[operation_index];
                    const auto projection = piece.projection;
                    const auto projection_index = projectionIndex(projection);
                    const auto &source_participant = *config_.initial_snapshot->placementOwnerMap().participantForId(piece.source_participant);
                    const auto &destination_participant = *config_.initial_snapshot->placementOwnerMap().participantForId(piece.destination_participant);
                    const bool source_local = is_local_participant(piece.source_participant);
                    const bool destination_local = is_local_participant(piece.destination_participant);
                    const bool crosses_world_rank = remote_transport && source_participant.world_rank != destination_participant.world_rank;
                    prepared.operations[operation_index] = {.projection = projection,
                        .destination_participant = piece.destination_participant, .source_participant = piece.source_participant};
                    prepared.operations[operation_index].requires_local_measurement = source_local || destination_local;
                    prepared.operations[operation_index].observer_world_rank = source_local || destination_local
                        ? config_.memory_authority->worldRank() : -1;
                    const auto destination_arrival = destination_local ? destination_arrivals[migration_index].at(piece.destination_participant) : nullptr;
                    DestinationProjectionGroup *const destination_group = destination_local
                        ? &destination_groups[migration_index].at(piece.destination_participant)[projection_index] : nullptr;
                    const auto completion_piece = static_cast<size_t>(std::count_if(pieces.begin(), pieces.begin() + operation_index,
                        [&](const auto &previous) { return previous.projection == projection &&
                            previous.destination_participant == piece.destination_participant; }));
                    std::optional<MoEOverlayPreparedExpertPayload> device_source_payload;
                    std::optional<MoEOverlayPreparedFragmentPayload> device_source_fragment;
                    const auto source_engine_for_role = [&]() -> std::shared_ptr<ITensorGemm>
                    {
                        if (!source_local) throw std::logic_error("A nonlocal physical source has no local engine permit");
                        if (!device_batch)
                            return source_members[migration_index].at(piece.source_participant)->engines[projection_index];
                        std::string source_error;
                        const MoEOverlayDevicePhysicalSlotKey source_key{
                            .participant_id = piece.source_participant, .layer_idx = migration.layer_idx,
                            .expert_id = migration.expert_id};
                        if (migration.source.shard_cohort)
                        {
                            device_source_fragment = impl_->device_slot_ledger->sourceFragmentPayload(
                                *device_batch, source_key, &source_error);
                            if (!device_source_fragment) throw std::invalid_argument(source_error);
                            switch (projection)
                            {
                            case ExpertTierWeightProjection::Gate: return device_source_fragment->gate();
                            case ExpertTierWeightProjection::Up: return device_source_fragment->up();
                            case ExpertTierWeightProjection::Down: return device_source_fragment->down();
                            }
                            throw std::invalid_argument("Device fragment source received an unknown projection");
                        }
                        device_source_payload = impl_->device_slot_ledger->sourcePayload(*device_batch,
                            source_key, &source_error);
                        if (!device_source_payload) throw std::invalid_argument(source_error);
                        return projectionEngine(*device_source_payload, projection);
                    };
                    if (!crosses_world_rank && !source_local)
                    {
                        prepared.operations[operation_index].operation = std::make_unique<ReadyPhysicalProjectionOperation>();
                        continue;
                    }

                    if (crosses_world_rank)
                    {
                        if (!remote_execution_fingerprint)
                            throw std::logic_error(
                                "ExpertOverlay remote projection lost transaction fingerprint ownership");

                        const auto identity = remoteProjectionIdentity(
                            expected_epoch,
                            candidate_epoch,
                            *remote_execution_fingerprint,
                            migrations,
                            migration_index, piece,
                            config_.initial_snapshot->placementOwnerMap());
                        MoEOverlayMPIRemoteProjectionBinding binding{
                            // Complete-wave ordinal is identical on source,
                            // receiver and uninvolved ranks, before any local filtering.
                            .lane_index = remote_piece_ordinals.by_migration.at(migration_index).at(operation_index).value(),
                            .identity = identity,
                            .purpose = purpose,
                        };

                        std::shared_ptr<MoEOverlayPreparedProjectionCompletion> remote_completion;
                        if (source_local)
                        {
                            auto source_engine = source_engine_for_role();
                            MoEOverlayPreparedWeightSource source;
                            std::string source_error;
                            if (!resolveMoEOverlayPreparedWeightSource(
                                    source_engine,
                                    source_participant.device,
                                    source,
                                    &source_error))
                            {
                                throw std::runtime_error(
                                    source_error.empty()
                                        ? "ExpertOverlay remote source does not expose prepared weights"
                                        : std::move(source_error));
                            }

                            if (source_participant.device.is_cpu())
                            {
                                if (source.format.isFloating())
                                {
                                    const auto manifest =
                                        makeMoEOverlayRemoteFloatingProjectionManifest(
                                            identity,
                                            source.floating,
                                            static_cast<std::uint32_t>(
                                                config_.staging_capacity_bytes)).forFloatingIntersection(piece.intersection);
                                    binding.source = std::make_shared<
                                        MoEOverlayHostRemoteProjectionSource>(
                                        manifest,
                                        remoteContiguousSourceRegions(
                                            std::span<const std::uint8_t>(
                                                static_cast<const std::uint8_t *>(
                                                    source.floating.data),
                                                source.floating.bytes)),
                                        source_engine);
                                    if (destination_participant.device.is_cpu())
                                        ++local_remote_cpu_operations;
                                    else
                                        ++local_remote_gpu_cpu_operations;
                                }
                                else if (!source.cpu_packed)
                                {
                                    throw std::runtime_error(
                                        "ExpertOverlay remote CPU source lost final NativeQuantised storage");
                                }
                                else if (destination_participant.device.is_cpu())
                                {
                                    const auto manifest =
                                        makeMoEOverlayRemoteCpuProjectionManifest(
                                            identity,
                                            *source.cpu_packed,
                                            static_cast<std::uint32_t>(
                                                config_.staging_capacity_bytes)).forCPUIntersection(piece.intersection);
                                    binding.source = std::make_shared<
                                        MoEOverlayHostRemoteProjectionSource>(
                                        manifest,
                                        remoteCpuSourceRegions(
                                            *source.cpu_packed),
                                        source_engine);
                                    ++local_remote_cpu_operations;
                                }
                                else
                                {
                                    const auto probe =
                                        makeCpuToGpuExpertTierWeightStreamManifest(
                                            *source.cpu_packed,
                                            candidate_epoch,
                                            migration.layer_idx,
                                            migration.expert_id,
                                            projection,
                                            1).forIntersection(piece.intersection);
                                    const auto maximum_units =
                                        maximumUnitsPerChunk(
                                            probe,
                                            config_.staging_capacity_bytes);
                                    const auto stream =
                                        makeCpuToGpuExpertTierWeightStreamManifest(
                                            *source.cpu_packed,
                                            candidate_epoch,
                                            migration.layer_idx,
                                            migration.expert_id,
                                            projection,
                                            maximum_units).forIntersection(piece.intersection);
                                    const auto manifest =
                                        makeMoEOverlayRemoteCpuProjectionManifest(
                                            identity,
                                            stream,
                                            static_cast<std::uint32_t>(
                                                config_.staging_capacity_bytes));
                                    binding.source = std::make_shared<
                                        MoEOverlayHostRemoteProjectionSource>(
                                        manifest,
                                        remoteCpuSourceRegions(
                                            *source.cpu_packed),
                                        source_engine);
                                    ++local_remote_gpu_cpu_operations;
                                }
                            }
                            else
                            {
                                const RemoteGpuLaneKey lane_key{
                                    .device = source_participant.device,
                                    .projection = projection,
                                    .role = RemoteGpuLaneRole::Source,
                                };
                                const auto lane =
                                    impl_->remote_gpu_lanes.find(lane_key);
                                if (lane == impl_->remote_gpu_lanes.end())
                                    throw std::runtime_error(
                                        "ExpertOverlay remote GPU source has no pre-materialized lane");
                                const std::size_t lane_index =
                                    remote_gpu_lane_cursor[lane_key]++;
                                if (lane_index >= lane->second.size())
                                    throw std::logic_error(
                                        "ExpertOverlay remote GPU source exceeded its preflighted parallel lane pool");

                                MoEOverlayRemoteProjectionManifest manifest;
                                if (source.format.isFloating())
                                {
                                    manifest =
                                        makeMoEOverlayRemoteFloatingProjectionManifest(
                                            identity,
                                            source.floating,
                                            static_cast<std::uint32_t>(
                                                config_.staging_capacity_bytes)).forFloatingIntersection(piece.intersection);
                                    if (destination_participant.device.is_cpu())
                                        ++local_remote_gpu_cpu_operations;
                                    else
                                        ++local_remote_gpu_blob_operations;
                                }
                                else if (destination_participant.device.is_cpu())
                                {
                                    const auto *source_format =
                                        native_quantised_formats::forSourceIdentity(
                                            source.format.native_quantised
                                                .codebook_id,
                                            source.format.native_quantised
                                                .is_superblock);
                                    if (!source_format)
                                        throw std::runtime_error(
                                            "ExpertOverlay remote GPU source lost catalogued provenance");
                                    const auto probe =
                                        makeGpuToCpuExpertTierWeightStreamManifest(
                                            *source_format,
                                            source.gpu_packed,
                                            expected_epoch,
                                            migration.layer_idx,
                                            migration.expert_id,
                                            projection,
                                            1).forIntersection(piece.intersection);
                                    const auto maximum_units =
                                        maximumUnitsPerChunk(
                                            probe,
                                            config_.staging_capacity_bytes);
                                    const auto stream =
                                        makeGpuToCpuExpertTierWeightStreamManifest(
                                            *source_format,
                                            source.gpu_packed,
                                            expected_epoch,
                                            migration.layer_idx,
                                            migration.expert_id,
                                            projection,
                                            maximum_units).forIntersection(piece.intersection);
                                    manifest =
                                        makeMoEOverlayRemoteCpuProjectionManifest(
                                            identity,
                                            stream,
                                            static_cast<std::uint32_t>(
                                                config_.staging_capacity_bytes));
                                    ++local_remote_gpu_cpu_operations;
                                }
                                else
                                {
                                    manifest =
                                        makeMoEOverlayRemoteGpuProjectionManifest(
                                            identity,
                                            source.gpu_packed,
                                            source.format.native_quantised,
                                            static_cast<std::uint32_t>(
                                                config_.staging_capacity_bytes)).forGPUIntersection(piece.intersection);
                                    ++local_remote_gpu_blob_operations;
                                }
                                const auto readiness =
                                    ExpertTierSourceReadiness::
                                        publishedResidencyBank(
                                            expected_epoch);
                                if (source.format.isFloating())
                                {
                                    binding.source = std::make_shared<
                                        MoEOverlayGpuRemoteProjectionSource>(
                                        std::move(manifest),
                                        lane->second[lane_index],
                                        source.floating,
                                        readiness,
                                        source_engine);
                                }
                                else
                                {
                                    binding.source = std::make_shared<
                                        MoEOverlayGpuRemoteProjectionSource>(
                                        std::move(manifest),
                                        lane->second[lane_index],
                                        source.gpu_packed,
                                        readiness,
                                        source_engine);
                                }
                            }
                        }

                        if (destination_local)
                        {
                            if (destination_participant.device.is_cpu())
                            {
                                auto &lease =
                                    *destinations[migration_index].at(piece.destination_participant).cpu;
                                const auto &destination = cpuProjection(
                                    lease, projection);
                                auto destination_engine = destination.engine;
                                ContiguousFloatingPointWeightDescriptor
                                    destination_floating;
                                const bool destination_is_floating =
                                    destination_engine
                                        ->exportContiguousFloatingPointWeights(
                                            destination_floating);
                                const auto *destination_packed =
                                    destination_engine
                                        ->exportCPUNativeQuantisedPackedWeights();
                                if (destination_is_floating ==
                                    (destination_packed != nullptr))
                                {
                                    throw std::runtime_error(
                                        "ExpertOverlay remote CPU destination must expose exactly one physical representation");
                                }

                                MoEOverlayRemoteProjectionManifest
                                    expected_manifest;
                                std::array<
                                    std::span<std::uint8_t>,
                                    kMoEOverlayRemoteProjectionRegionCount>
                                    destination_regions;
                                auto match_policy =
                                    MoEOverlayRemoteProjectionManifestMatchPolicy::
                                        Exact;
                                if (destination_is_floating)
                                {
                                    if (!destination_floating.valid() ||
                                        destination.destination_bytes.data() !=
                                            destination_floating.data ||
                                        destination.destination_bytes.size() !=
                                            destination_floating.bytes)
                                    {
                                        throw std::runtime_error(
                                            "ExpertOverlay remote floating CPU destination disagrees with its final slot");
                                    }
                                    expected_manifest =
                                        makeMoEOverlayRemoteFloatingDestinationProjectionManifest(
                                            identity,
                                            destination_floating, piece.intersection,
                                            static_cast<std::uint32_t>(
                                                config_.staging_capacity_bytes));
                                    destination_regions =
                                        remoteContiguousDestinationRegions(
                                            destination.destination_bytes);
                                    if (source_participant.device.is_cpu())
                                        ++local_remote_cpu_operations;
                                    else
                                        ++local_remote_gpu_cpu_operations;
                                }
                                else if (source_participant.device.is_cpu())
                                {
                                    expected_manifest =
                                        makeMoEOverlayRemoteCpuDestinationProjectionManifest(
                                            identity, *destination_packed, piece.intersection,
                                            static_cast<std::uint32_t>(config_.staging_capacity_bytes));
                                    destination_regions =
                                        remoteCpuDestinationRegions(
                                            destination.destination_bytes);
                                    ++local_remote_cpu_operations;
                                }
                                else
                                {
                                    const auto *source_format =
                                        native_quantised_formats::forSourceIdentity(
                                            destination_packed->codebook_id,
                                            destination_packed->is_superblock);
                                    if (!source_format)
                                        throw std::runtime_error(
                                            "ExpertOverlay remote CPU destination lost source provenance");
                                    const auto probe =
                                        makeGpuToCpuExpertTierWeightStreamManifest(
                                            *source_format,
                                            piece.intersection.source().rows,
                                            piece.intersection.source().columns,
                                            expected_epoch,
                                            migration.layer_idx,
                                            migration.expert_id,
                                            projection,
                                            1).forIntersection(piece.intersection);
                                    const auto maximum_units =
                                        maximumUnitsPerChunk(
                                            probe,
                                            config_.staging_capacity_bytes);
                                    const auto stream =
                                        makeGpuToCpuExpertTierWeightStreamManifest(
                                            *source_format,
                                            piece.intersection.source().rows,
                                            piece.intersection.source().columns,
                                            expected_epoch,
                                            migration.layer_idx,
                                            migration.expert_id,
                                            projection,
                                            maximum_units).forIntersection(piece.intersection);
                                    expected_manifest =
                                        makeMoEOverlayRemoteCpuProjectionManifest(
                                            identity,
                                            stream,
                                            static_cast<std::uint32_t>(
                                                config_.staging_capacity_bytes));
                                    match_policy =
                                        MoEOverlayRemoteProjectionManifestMatchPolicy::
                                            EquivalentGpuSourceForFinalCpuStorage;
                                    destination_regions =
                                        remoteCpuDestinationRegions(
                                            destination.destination_bytes);
                                    ++local_remote_gpu_cpu_operations;
                                }

                                auto storage = std::make_shared<
                                    MoEOverlayHostRemoteProjectionDestination>(
                                    identity,
                                    destination_regions,
                                    destination_engine,
                                    expected_manifest,
                                    match_policy, destination_group->storage->encoded_write_mutex);
                                auto finalized =
                                    std::make_shared<FinalizingRemoteProjectionDestination>(
                                        std::move(storage), std::move(destination_engine));
                                binding.destination = finalized;
                                const auto rows = destination_is_floating ? destination_floating.n
                                                                          : destination_packed->N;
                                const auto columns = destination_is_floating
                                                         ? destination_floating.k
                                                         : destination_packed->K;
                                remote_completion = destination_group->completion;
                            }
                            else
                            {
                                auto &lease =
                                    *destinations[migration_index].at(piece.destination_participant).gpu;
                                const auto destination_slot = gpuProjection(
                                    lease, projection);
                                const auto slot_lifetime = lease.lifetime;
                                const auto destination_format =
                                    destination_slot.spec.format;
                                const auto destination_device =
                                    destination_participant.device;
                                const RemoteGpuLaneKey lane_key{
                                    .device = destination_device,
                                    .projection = projection,
                                    .role = RemoteGpuLaneRole::Destination,
                                };
                                const auto lane =
                                    impl_->remote_gpu_lanes.find(lane_key);
                                if (lane == impl_->remote_gpu_lanes.end())
                                    throw std::runtime_error(
                                        "ExpertOverlay remote GPU destination has no pre-materialized lane");
                                const std::size_t lane_index =
                                    remote_gpu_lane_cursor[lane_key]++;
                                if (lane_index >= lane->second.size())
                                    throw std::logic_error(
                                        "ExpertOverlay remote GPU destination exceeded its preflighted parallel lane pool");

                                MoEOverlayGpuRemoteProjectionDestinationFactory
                                    factory =
                                        [destination_slot,
                                         destination_device,
                                         slot_lifetime,
                                         destination_format,
                                         intersection = piece.intersection,
                                         role_storage = destination_group->storage](
                                            const MoEOverlayRemoteProjectionManifest &
                                                manifest,
                                            MoEOverlayGpuRemoteProjectionDestinationBinding *
                                                output,
                                            std::string *error) -> bool
                                {
                                    if (!output || !manifest.matchesGpuDestinationProjection(
                                            intersection, destination_format,
                                            destination_slot.spec.N, destination_slot.spec.K))
                                    {
                                        if (error) *error = destination_format.isFloating()
                                            ? "Remote floating ExpertOverlay GPU arrival differs from the model projection contract"
                                            : "Remote ExpertOverlay GPU arrival differs from the model projection contract";
                                        return false;
                                    }
                                    if (destination_format.isFloating())
                                    {
                                        auto descriptor =
                                            makeGpuFloatingDestinationDescriptor(
                                                destination_slot,
                                                destination_format);
                                        auto engine = retainFloatingGpuDestination(*role_storage,
                                            destination_device, descriptor, slot_lifetime);
                                        *output = {
                                            .floating_descriptor = descriptor,
                                            .engine = std::move(engine),
                                        };
                                        if (error)
                                            error->clear();
                                        return true;
                                    }

                                    const auto source_identity =
                                        destination_format.native_quantised;
                                    auto descriptor =
                                        makeGpuDestinationDescriptor(
                                            destination_slot,
                                            destination_slot.spec.N,
                                            destination_slot.spec.K,
                                            manifest.gpu_codebook_id,
                                            manifest
                                                .gpu_payload_bytes_per_block,
                                            manifest.gpu_is_asymmetric != 0,
                                            manifest.gpu_has_emins != 0,
                                            source_identity);
                                    auto engine = retainNativeGpuDestination(*role_storage,
                                        destination_device, descriptor, slot_lifetime, source_identity);
                                    *output = {
                                        .descriptor = descriptor,
                                        .engine = std::move(engine),
                                    };
                                    if (error)
                                        error->clear();
                                    return true;
                                };
                                auto gpu_storage = std::make_shared<
                                    MoEOverlayGpuRemoteProjectionDestination>(
                                    identity,
                                    lane->second[lane_index],
                                    std::move(factory),
                                    slot_lifetime);
                                auto finalized =
                                    std::make_shared<FinalizingRemoteProjectionDestination>(
                                        gpu_storage,
                                        [gpu_storage]() { return gpu_storage->preparedEngine(); });
                                binding.destination = finalized;
                                remote_completion = destination_group->completion;
                                if (source_participant.device.is_cpu())
                                    ++local_remote_gpu_cpu_operations;
                                else
                                    ++local_remote_gpu_blob_operations;
                            }
                        }

                        remote_targets.push_back({
                            .migration_index = migration_index,
                            .operation_index = operation_index,
                            .piece_index = completion_piece,
                            .completion = std::move(remote_completion),
                        });
                        remote_bindings.push_back(std::move(binding));
                        continue;
                    }

                    auto source_engine = source_engine_for_role();
                    MoEOverlayPreparedWeightSource source;
                    std::string source_error;
                    if (!resolveMoEOverlayPreparedWeightSource(
                            source_engine,
                            source_participant.device,
                            source,
                            &source_error))
                    {
                        throw std::runtime_error(source_error);
                    }

                    std::shared_ptr<ITensorGemm> destination_engine;
                    std::unique_ptr<IMoEOverlayTierTransferOperation>
                        physical_operation;
                    const EdgeKey edge{
                        .source = source_participant.device,
                        .destination = destination_participant.device,
                        .projection = projection,
                    };
                    std::shared_ptr<SharedCpuCopyLanePool> cpu_copy_lane;
                    if (source_participant.device.is_cpu() &&
                        destination_participant.device.is_cpu())
                    {
                        const CpuAddressEdgeKey cpu_edge{
                            .source = source_participant.address,
                            .destination = destination_participant.address,
                            .projection = projection,
                        };
                        const auto found = impl_->cpu_copy_lanes.find(cpu_edge);
                        if (found == impl_->cpu_copy_lanes.end())
                            throw std::runtime_error(
                                "ExpertOverlay CPU address edge has no persistent parallel worker pool");
                        cpu_copy_lane = found->second;
                    }

                    if (destination_participant.device.is_cpu())
                    {
                        auto &lease = *destinations[migration_index].at(piece.destination_participant).cpu;
                        const auto &destination = cpuProjection(
                            lease, projection);
                        destination_engine = destination.engine;
                        if (source.format.isFloating())
                        {
                            ContiguousFloatingPointWeightDescriptor
                                destination_floating;
                            if (!destination_engine
                                     ->exportContiguousFloatingPointWeights(
                                         destination_floating) ||
                                !destination_floating.valid() ||
                                source.floating.type !=
                                    destination_floating.type ||
                                source.floating.n != piece.intersection.source().rows ||
                                source.floating.k != piece.intersection.source().columns ||
                                destination_floating.n != piece.intersection.destination().rows ||
                                destination_floating.k != piece.intersection.destination().columns ||
                                destination.destination_bytes.data() !=
                                    destination_floating.data)
                            {
                                throw std::runtime_error(
                                    "ExpertOverlay CPU floating destination disagrees with the model projection contract");
                            }
                            if (source_participant.device.is_cpu())
                            {
                                physical_operation =
                                    std::make_unique<AdmittedCpuCopyOperation>(
                                        cpu_copy_lane,
                                        std::span<const std::uint8_t>(
                                            static_cast<const std::uint8_t *>(
                                                source.floating.data),
                                            source.floating.bytes),
                                        destination.destination_bytes,
                                        cpu_copy_wave_gate,
                                        source_engine,
                                        destination_engine,
                                        FloatingCPUCopy{piece.intersection,
                                            ExpertWeightFormat::floating(source.floating.type).floatingElementBytes()});
                                ++local_cpu_copy_operations;
                            }
                            else
                            {
                                const auto lane = impl_->weight_lanes.find(edge);
                                if (lane == impl_->weight_lanes.end())
                                    throw std::runtime_error(
                                        "ExpertOverlay floating GPU-to-CPU edge has no persistent lane");
                                physical_operation =
                                    std::make_unique<AdmittedWeightOperation>(
                                        lane->second,
                                        AdmittedWeightOperation::Direction::
                                            GpuToCpuContiguous,
                                        source.floating,
                                        destination.destination_bytes,
                                        std::span<const std::uint8_t>{},
                                        ExpertTierSourceReadiness::
                                            publishedResidencyBank(
                                                expected_epoch),
                                        source_engine,
                                        destination_engine, piece.intersection,
                                        wave_timing_capture);
                                ++local_gpu_cpu_operations;
                            }
                        }
                        else
                        {
                        const auto *destination_packed =
                            destination_engine
                                ->exportCPUNativeQuantisedPackedWeights();
                        if (!destination_packed)
                            throw std::runtime_error(
                                "ExpertOverlay CPU destination engine cannot export final storage");

                        if (source_participant.device.is_cpu())
                        {
                            if (source.cpu_packed->codebook_id != destination_packed->codebook_id ||
                                source.cpu_packed->is_superblock != destination_packed->is_superblock ||
                                source.cpu_packed->encoding != destination_packed->encoding)
                                throw std::invalid_argument("CPU rectangle movement cannot change original native format");
                            physical_operation =
                                std::make_unique<AdmittedCpuCopyOperation>(
                                    cpu_copy_lane,
                                    std::span<const std::uint8_t>(
                                        source.cpu_packed
                                            ->native_interleaved.data(),
                                        source.cpu_packed
                                            ->native_interleaved.size()),
                                    destination.destination_bytes,
                                    cpu_copy_wave_gate,
                                    source_engine,
                                    destination_engine,
                                    nativeCPUCopyGeometry(*source.cpu_packed, *destination_packed,
                                        piece.intersection, config_.staging_capacity_bytes,
                                        destination_group->storage->encoded_write_mutex));
                            ++local_cpu_copy_operations;
                        }
                        else
                        {
                            const auto *source_format =
                                native_quantised_formats::forSourceIdentity(
                                    source.format.native_quantised.codebook_id,
                                    source.format.native_quantised.is_superblock);
                            if (!source_format)
                                throw std::runtime_error(
                                    "ExpertOverlay GPU source lost catalogued provenance");
                            const auto probe =
                                makeGpuToCpuExpertTierWeightStreamManifest(
                                    *source_format,
                                    source.gpu_packed,
                                    expected_epoch,
                                    migration.layer_idx,
                                    migration.expert_id,
                                    projection,
                                    1).forIntersection(piece.intersection);
                            const auto maximum_units = maximumUnitsPerChunk(
                                probe,
                                config_.staging_capacity_bytes);
                            const auto manifest =
                                makeGpuToCpuExpertTierWeightStreamManifest(
                                    *source_format,
                                    source.gpu_packed,
                                    expected_epoch,
                                    migration.layer_idx,
                                    migration.expert_id,
                                    projection,
                                    maximum_units).forIntersection(piece.intersection);
                            const auto lane = impl_->weight_lanes.find(edge);
                            if (lane == impl_->weight_lanes.end())
                                throw std::runtime_error(
                                    "ExpertOverlay GPU-to-CPU edge has no persistent lane");
                            physical_operation =
                                std::make_unique<AdmittedWeightOperation>(
                                    lane->second,
                                    AdmittedWeightOperation::Direction::
                                        GpuToCpuRepacked,
                                    manifest.deviceLayout(),
                                    gpuConstView(source.gpu_packed),
                                    destination.destination_bytes,
                                    std::span<const std::uint8_t>{},
                                    ExpertTierGpuMutableProjectionView{},
                                    ExpertTierSourceReadiness::publishedResidencyBank(
                                        expected_epoch),
                                    source_engine,
                                    destination_engine, destination_group->storage->encoded_write_mutex,
                                    wave_timing_capture);
                            ++local_gpu_cpu_operations;
                        }
                        }
                    }
                    else
                    {
                        auto &lease = *destinations[migration_index].at(piece.destination_participant).gpu;
                        const auto &destination_slot = gpuProjection(
                            lease, projection);
                        GpuExpertPackedDescriptor destination_descriptor;

                        if (source.format.isFloating())
                        {
                            const auto destination_floating =
                                makeGpuFloatingDestinationDescriptor(
                                    destination_slot,
                                    source.format);
                            destination_engine = retainFloatingGpuDestination(*destination_group->storage,
                                destination_participant.device, destination_floating, lease.lifetime);
                            if (source_participant.device.is_cpu())
                            {
                                const auto cpu_source =
                                    std::span<const std::uint8_t>(
                                        static_cast<const std::uint8_t *>(
                                            source.floating.data),
                                        source.floating.bytes);
                                const auto lane = impl_->weight_lanes.find(edge);
                                if (lane == impl_->weight_lanes.end())
                                    throw std::runtime_error(
                                        "ExpertOverlay floating CPU-to-GPU edge has no persistent lane");
                                physical_operation =
                                    std::make_unique<AdmittedWeightOperation>(
                                        lane->second,
                                        AdmittedWeightOperation::Direction::
                                            CpuToGpuContiguous,
                                        destination_floating,
                                        std::span<std::uint8_t>{},
                                        cpu_source,
                                        ExpertTierSourceReadiness::
                                            publishedResidencyBank(
                                                expected_epoch),
                                        source_engine,
                                        destination_engine, piece.intersection,
                                        wave_timing_capture);
                                ++local_gpu_cpu_operations;
                            }
                            else if (const auto peer_lane =
                                    impl_->peer_lanes.find(edge);
                                peer_lane != impl_->peer_lanes.end())
                            {
                                physical_operation =
                                    std::make_unique<AdmittedPeerOperation>(
                                        peer_lane->second,
                                        source.floating,
                                        destination_floating,
                                        ExpertTierSourceReadiness::publishedResidencyBank(
                                            expected_epoch),
                                        source_engine,
                                        destination_engine, piece.intersection,
                                        wave_timing_capture);
                                ++local_same_backend_gpu_operations;
                            }
                            else
                            {
                                const auto blob_lane =
                                    impl_->blob_lanes.find(edge);
                                if (blob_lane == impl_->blob_lanes.end())
                                    throw std::runtime_error(
                                        "ExpertOverlay floating GPU edge has no persistent direct or host-relay lane");
                                physical_operation =
                                    std::make_unique<AdmittedBlobOperation>(
                                        blob_lane->second,
                                        source.floating,
                                        destination_floating,
                                        ExpertTierSourceReadiness::publishedResidencyBank(
                                            expected_epoch),
                                        source_engine,
                                        destination_engine, piece.intersection);
                                if (source_participant.device.type ==
                                    destination_participant.device.type)
                                {
                                    ++local_same_backend_gpu_operations;
                                }
                                else
                                {
                                    ++local_heterogeneous_gpu_operations;
                                }
                            }
                        }
                        else if (source_participant.device.is_cpu())
                        {
                            const auto probe =
                                makeCpuToGpuExpertTierWeightStreamManifest(
                                    *source.cpu_packed,
                                    candidate_epoch,
                                    migration.layer_idx,
                                    migration.expert_id,
                                    projection,
                                    1).forIntersection(piece.intersection);
                            const auto maximum_units = maximumUnitsPerChunk(
                                probe,
                                config_.staging_capacity_bytes);
                            const auto manifest =
                                makeCpuToGpuExpertTierWeightStreamManifest(
                                    *source.cpu_packed,
                                    candidate_epoch,
                                    migration.layer_idx,
                                    migration.expert_id,
                                    projection,
                                    maximum_units).forIntersection(piece.intersection);
                            const auto layout = manifest.deviceLayout();
                            destination_descriptor =
                                makeGpuDestinationDescriptor(
                                    destination_slot,
                                    layout.gpu_N,
                                    layout.gpu_K,
                                    layout.gpu_codebook_id,
                                    layout.gpu_payload_bytes_per_block,
                                    layout.gpu_is_asymmetric != 0,
                                    layout.gpu_has_emins != 0,
                                    source.format.native_quantised);
                            destination_engine = retainNativeGpuDestination(*destination_group->storage,
                                destination_participant.device, destination_descriptor, lease.lifetime,
                                source.format.native_quantised);
                            const auto lane = impl_->weight_lanes.find(edge);
                            if (lane == impl_->weight_lanes.end())
                                throw std::runtime_error(
                                    "ExpertOverlay CPU-to-GPU edge has no persistent lane");
                            physical_operation =
                                std::make_unique<AdmittedWeightOperation>(
                                    lane->second,
                                    AdmittedWeightOperation::Direction::
                                        CpuToGpuRepacked,
                                    layout,
                                    ExpertTierGpuConstProjectionView{},
                                    std::span<std::uint8_t>{},
                                    std::span<const std::uint8_t>(
                                        source.cpu_packed
                                            ->native_interleaved.data(),
                                        source.cpu_packed
                                            ->native_interleaved.size()),
                                    gpuMutableView(destination_descriptor),
                                    ExpertTierSourceReadiness::publishedResidencyBank(
                                        expected_epoch),
                                    source_engine,
                                    destination_engine,
                                    std::shared_ptr<std::mutex>{},
                                    wave_timing_capture);
                            ++local_gpu_cpu_operations;
                        }
                        else
                        {
                            destination_descriptor =
                                makeGpuDestinationDescriptor(
                                    destination_slot,
                                    destination_slot.spec.N,
                                    destination_slot.spec.K,
                                    source.gpu_packed.codebook_id,
                                    source.gpu_packed
                                        .payload_bytes_per_block,
                                    source.gpu_packed.is_asymmetric,
                                    source.gpu_packed.has_emins,
                                    source.format.native_quantised);
                            destination_engine = retainNativeGpuDestination(*destination_group->storage,
                                destination_participant.device, destination_descriptor, lease.lifetime,
                                source.format.native_quantised);

                            if (const auto peer_lane =
                                    impl_->peer_lanes.find(edge);
                                peer_lane != impl_->peer_lanes.end())
                            {
                                physical_operation =
                                    std::make_unique<AdmittedPeerOperation>(
                                        peer_lane->second,
                                        source.gpu_packed,
                                        destination_descriptor,
                                        ExpertTierSourceReadiness::publishedResidencyBank(
                                            expected_epoch),
                                        source_engine,
                                        destination_engine, piece.intersection,
                                        wave_timing_capture);
                                ++local_same_backend_gpu_operations;
                            }
                            else
                            {
                                const auto blob_lane =
                                    impl_->blob_lanes.find(edge);
                                if (blob_lane == impl_->blob_lanes.end())
                                    throw std::runtime_error(
                                        "ExpertOverlay GPU edge has no persistent direct or host-relay lane");
                                physical_operation =
                                    std::make_unique<AdmittedBlobOperation>(
                                        blob_lane->second,
                                        source.gpu_packed,
                                        destination_descriptor,
                                        ExpertTierSourceReadiness::publishedResidencyBank(
                                            expected_epoch),
                                        source_engine,
                                        destination_engine, piece.intersection);
                                if (source_participant.device.type ==
                                    destination_participant.device.type)
                                {
                                    ++local_same_backend_gpu_operations;
                                }
                                else
                                {
                                    ++local_heterogeneous_gpu_operations;
                                }
                            }
                        }
                    }

                    if (!destination_group) throw std::logic_error("Local physical destination lost its original role completion");
                    if (destination_group->storage->engine && destination_group->storage->engine.get() != destination_engine.get())
                        throw std::logic_error("Physical pieces attempted to replace an already reserved final role engine");
                    destination_group->storage->engine = destination_engine;
                    prepared.operations[operation_index].operation =
                        std::make_unique<MoEOverlayPreparedProjectionOperation>(
                            std::move(physical_operation), destination_group->completion, completion_piece);
                }
            }

            if (!remote_bindings.empty())
            {
                auto remote_wave = remote_transport->reserveWave(
                    std::move(remote_bindings));
                if (remote_wave.status ==
                    MoEOverlayResidencyStageStartStatus::Deferred)
                {
                    result.migrations.clear();
                    result.status = remote_wave.status;
                    result.error = std::move(remote_wave.error);
                    impl_->waves_deferred.fetch_add(
                        1, std::memory_order_relaxed);
                    return result;
                }
                if (remote_wave.status !=
                        MoEOverlayResidencyStageStartStatus::Started ||
                    remote_wave.operations.size() != remote_targets.size())
                {
                    result.migrations.clear();
                    return fail(
                        remote_wave.error.empty()
                            ? "ExpertOverlay remote projection transport returned an invalid wave shape"
                            : std::move(remote_wave.error));
                }
                for (std::size_t remote_index = 0;
                     remote_index < remote_targets.size();
                     ++remote_index)
                {
                    const auto &target = remote_targets[remote_index];
                    auto operation = std::move(remote_wave.operations[remote_index]);
                    if (target.completion)
                        operation = std::make_unique<MoEOverlayPreparedProjectionOperation>(
                            std::move(operation), target.completion, target.piece_index);
                    result.migrations[target.migration_index].operations[target.operation_index].operation =
                        std::move(operation);
                }
            }
        }
        catch (const std::exception &error)
        {
            return fail(error.what());
        }

        // Every constructor and reserveWave above is an inert reservation:
        // CPU workers, GPU lanes and MPI requests start only on first poll,
        // after the caller's complete-wave consensus. A setup failure therefore
        // returns an unsubmitted cleanup prefix and can release local bank
        // readers without any active copy. Only Started work escapes this
        // scope and must retain its exact bank through Ready or abort drain.
        try
        {
        if (!device_batch)
            for (size_t index = 0; index < result.migrations.size(); ++index)
                for (auto &piece : result.migrations[index].operations)
                {
                    const auto source = source_members[index].find(piece.source_participant);
                    if (source != source_members[index].end())
                        piece.operation = std::make_unique<SourceMemberRetainedOperation>(
                            std::move(piece.operation), source->second);
                }

        }
        catch (const std::exception &error)
        {
            return fail(error.what());
        }

        size_t actual_projection_operations = 0;
        for (const auto &pieces : physical_pieces)
        {
            if (pieces.size() > std::numeric_limits<size_t>::max() - actual_projection_operations)
                return fail("Physical expert piece population overflows size_t");
            actual_projection_operations += pieces.size();
        }
        result.status = MoEOverlayResidencyStageStartStatus::Started;
        impl_->projection_operations_prepared.fetch_add(
            actual_projection_operations,
            std::memory_order_relaxed);
        impl_->cpu_copy_operations.fetch_add(
            local_cpu_copy_operations, std::memory_order_relaxed);
        impl_->remote_cpu_operations.fetch_add(
            local_remote_cpu_operations, std::memory_order_relaxed);
        impl_->remote_gpu_cpu_operations.fetch_add(
            local_remote_gpu_cpu_operations, std::memory_order_relaxed);
        impl_->remote_gpu_blob_operations.fetch_add(
            local_remote_gpu_blob_operations, std::memory_order_relaxed);
        impl_->gpu_cpu_operations.fetch_add(
            local_gpu_cpu_operations, std::memory_order_relaxed);
        impl_->same_backend_gpu_operations.fetch_add(
            local_same_backend_gpu_operations, std::memory_order_relaxed);
        impl_->heterogeneous_gpu_operations.fetch_add(
            local_heterogeneous_gpu_operations,
            std::memory_order_relaxed);
        impl_->waves_prepared.fetch_add(1, std::memory_order_relaxed);
        PerfStatsCollector::addCounter(
            "moe_overlay_residency",
            "physical_waves_prepared",
            1.0,
            "maintenance",
            config_.perf_device,
            {{"migrations", std::to_string(migrations.size())},
             {"projections",
              std::to_string(actual_projection_operations)},
             {"cpu_parallel_copy_operations",
              std::to_string(local_cpu_copy_operations)},
             {"remote_cpu_endpoints",
              std::to_string(local_remote_cpu_operations)},
             {"remote_gpu_cpu_endpoints",
              std::to_string(local_remote_gpu_cpu_operations)},
             {"remote_gpu_blob_endpoints",
              std::to_string(local_remote_gpu_blob_operations)}});
        return result;
    }

} // namespace llaminar2
