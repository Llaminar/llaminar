/**
 * @file MoEOverlayResidencySetup.cpp
 * @brief Materialize one compact expert runtime from its admitted owner geometry.
 *
 * Setup validates the complete PMA certificate and exact routed interval before
 * allocating transaction evidence. It uses the existing source manifest, owner
 * map, histogram, residency authority and preparation compiler. Pipeline offsets
 * never become local layer IDs in externally visible placement or evidence.
 */
#include "MoEOverlayResidencySetup.h"
#include "MoEOverlayResidencyAuthority.h"
#include "MoEOverlayParticipantResidency.h"
#include "MoEOverlayEconomyCalibrationPlanner.h"
#include "MoEOverlayInferenceInterferenceProbe.h"
#include "MoEExpertOverlayPreparationPlan.h"
#include "loaders/ModelLoader.h"
#include "utils/PerfStatsCollector.h"
#include <algorithm>
#include <limits>
#include <stdexcept>

namespace llaminar2
{
    MoEOverlayResidencySetup MoEOverlayResidencySetup::create(const MoEOverlayResidencySetupInput &input)
    {
        const auto &metadata = input.metadata;
        const auto &plan = input.plan;
        const auto &rebalance = input.rebalance;
        const auto &storage = input.storage_policy;
        const auto &capacity = input.capacity;
        if (metadata.first_model_layer < 0 || metadata.num_layers <= 0 || metadata.num_experts <= 0 ||
            metadata.num_layers > std::numeric_limits<int>::max() - metadata.first_model_layer ||
            metadata.main_inference_layer_count <= metadata.first_model_layer ||
            metadata.main_inference_layer_count > metadata.first_model_layer + metadata.num_layers ||
            !plan.usesExpertOverlayAuthority() || plan.placements.empty() ||
            plan.first_model_layer != metadata.first_model_layer ||
            plan.authority_execution != resolveMoEOverlayAuthorityExecutionKind(plan) ||
            input.world_size <= 0 || input.world_rank < 0 || input.world_rank >= input.world_size ||
            storage.overlay_world_size != input.world_size || !input.memory ||
            input.memory->worldRank() != input.world_rank ||
            !capacity.physical_memory_admission || input.memory->admission() != capacity.physical_memory_admission ||
            capacity.num_experts != metadata.num_experts || capacity.layer_footprints.size() != size_t(metadata.num_layers))
            throw std::invalid_argument("Expert runtime setup requires matching model, placement and physical admission owners");
        for (int row = 0; row < metadata.num_layers; ++row)
            if (capacity.layer_footprints[row].layer_idx != metadata.first_model_layer + row)
                throw std::invalid_argument("Expert runtime setup capacity belongs to another model layer interval");
        if (rebalance.mode != MoERebalanceRuntimeMode::Off && rebalance.mode != MoERebalanceRuntimeMode::Observe &&
            rebalance.mode != MoERebalanceRuntimeMode::Dynamic)
            throw std::invalid_argument("Expert runtime setup has an unknown maintenance mode");
        if (storage.migration_storage != MoEOverlayCapacityAdmission::selectMigrationStorage(plan, rebalance.mode) ||
            plan.replica_cache_capacity != capacity.replica_cache_capacity ||
            std::any_of(plan.routed_tiers.begin(), plan.routed_tiers.end(),
                [](const auto &tier) { return tier.resolved_live_experts_per_layer.empty(); }))
            throw std::invalid_argument("Expert runtime setup requires frozen quotas and the admitted migration policy");
        // This existing capacity validator proves the frozen placements still
        // satisfy every exact quota. The returned copy is never a new authority.
        (void)MoEOverlayCapacityResolver::installResolvedQuotas(plan, capacity);
        const bool dynamic = rebalance.mode == MoERebalanceRuntimeMode::Dynamic;
        const bool host_dynamic = dynamic && plan.authority_execution == MoEOverlayAuthorityExecutionKind::HostResident;
        const auto owner_map = MoEExpertOwnerMap::build(plan);
        owner_map.requireLayerGeometry(metadata.num_layers, metadata.num_experts, metadata.first_model_layer);
        const int top_k = input.loader.getInt(input.loader.architecture() + ".expert_used_count", 0);
        if (input.loader.getInt(input.loader.architecture() + ".expert_count", 0) != metadata.num_experts ||
            top_k <= 0 || top_k > metadata.num_experts)
            throw std::invalid_argument("Expert runtime setup source disagrees with routing geometry");

        std::vector<int> local_ids;
        for (const auto &participant : owner_map.participants())
        {
            if (!participant.world_rank_known || participant.world_rank < 0 || participant.world_rank >= input.world_size)
                throw std::invalid_argument("Expert runtime setup requires resolved participant ranks");
            const auto resource = std::find_if(capacity.physical_resources.begin(), capacity.physical_resources.end(),
                [&](const auto &candidate) {
                    return candidate.bom().resource().world_rank == participant.world_rank &&
                        candidate.device() == participant.device;
                });
            if (resource == capacity.physical_resources.end() || resource->authority() != capacity.physical_memory_admission)
                throw std::invalid_argument("Expert runtime setup participant is outside its stage's admitted resources");
            if (participant.world_rank == input.world_rank)
            {
                if (!input.memory->contains(participant.device))
                    throw std::invalid_argument("Expert runtime setup endpoint has no local physical allocator");
                local_ids.push_back(participant.participant_id);
            }
        }
        std::shared_ptr<const MoEOverlayEconomyCalibrationLayerCatalog> catalog;
        if (dynamic)
            catalog = std::make_shared<const MoEOverlayEconomyCalibrationLayerCatalog>(
                buildMoEOverlayLayerWeightManifestFromGGUF(input.loader.getModel(), metadata.num_layers,
                    metadata.num_experts, metadata.first_model_layer));

        MoEOverlayResidencySetup result;
        if (rebalance.mode != MoERebalanceRuntimeMode::Off)
        {
            const auto boundary = plan.lastPlacementLayerBefore(metadata.main_inference_layer_count);
            if (!boundary || *boundary < metadata.first_model_layer)
                throw std::invalid_argument("Expert runtime setup has no routed main layer for its token boundary");
            DecodeExpertHistogramConfig histogram{
                .num_layers = metadata.num_layers, .num_experts = metadata.num_experts, .top_k = top_k,
                .window_size = std::max(1, rebalance.window_size), .token_boundary_layer_idx = *boundary};
            for (const auto &participant : owner_map.participants()) histogram.sockets.push_back(participant.device);
            histogram.ownership = owner_map.layeredOwnership(metadata.num_layers, metadata.num_experts, metadata.first_model_layer);
            if (host_dynamic)
            {
                if (!capacity.host_demand_memory)
                    throw std::logic_error("Host Dynamic expert setup has no admitted transaction-demand geometry");
                histogram.transaction_demand = capacity.host_demand_memory->bind(histogram, input.memory);
            }
            result.histogram_ = std::make_shared<DecodeExpertHistogram>(std::move(histogram));
            if (dynamic) result.interference_ = std::make_shared<MoEOverlayInferenceInterferenceProbe>();
            PerfStatsCollector::addCounter("moe_overlay_residency", "histogram_token_boundary_configured", 1.0,
                "model_setup", {}, {{"boundary_layer", std::to_string(*boundary)},
                    {"first_model_layer", std::to_string(metadata.first_model_layer)},
                    {"main_layer_count", std::to_string(metadata.main_inference_layer_count - metadata.first_model_layer)},
                    {"resident_layer_capacity", std::to_string(metadata.num_layers)},
                    {"retained_auxiliary_layers", std::to_string(metadata.num_layers -
                        (metadata.main_inference_layer_count - metadata.first_model_layer))}});
        }
        std::string perf_device;
        for (const auto &tier : plan.routed_tiers)
            perf_device += (perf_device.empty() ? "" : "/") + tier.name;
        if (perf_device.empty()) perf_device = "expert_overlay";
        result.authority_ = std::make_shared<MoEOverlayResidencyAuthority>(MoEOverlayResidencyAuthority::Config{
            .initial_plan = plan, .model_metadata = metadata, .maintenance_mode = rebalance.mode,
            .histogram = result.histogram_.get(),
            .histogram_max_window_tokens = static_cast<std::uint64_t>(std::max(0, rebalance.max_window_size)),
            .histogram_window_growth_factor = static_cast<double>(rebalance.window_growth_factor),
            .economy_layer_catalog = std::move(catalog),
            .participant_rebalance_policy = {.enabled = host_dynamic,
                .imbalance_threshold_per_mille = rebalance.dynamic_imbalance_threshold_per_mille,
                .minimum_improvement_per_mille = rebalance.dynamic_min_improvement_per_mille,
                .maximum_swaps_per_layer = rebalance.dynamic_max_swaps_per_layer,
                .maximum_plan_entries_per_wave = rebalance.dynamic_max_plan_entries_per_wave,
                .minimum_window_activations = rebalance.dynamic_min_window_activations},
            .shadow_slots_per_endpoint_layer = storage.shadow_slots_per_endpoint_layer,
            .max_concurrent_cycles = rebalance.resolvedMigrationCyclesPerWave(), .perf_device = std::move(perf_device)});
        const auto snapshot = result.authority_->snapshot();
        if (!snapshot || !snapshot->valid())
            throw std::logic_error("Expert runtime setup did not publish a valid initial residency epoch");
        std::shared_ptr<const MoEExpertOverlayPreparationPlan> preparation;
        if (std::any_of(plan.domains.begin(), plan.domains.end(), [](const auto &domain) {
                return domain.routed_compute_policy == RoutedExpertComputePolicy::GateUpOwnedDownColumns;
            }))
        {
            const auto runtime = resolveMoEExpertOverlayRuntimePlan(snapshot->placement_plan,
                {.current_world_rank = input.world_rank, .validate_mvp_root_reachability = false});
            preparation = std::make_shared<const MoEExpertOverlayPreparationPlan>(
                MoEExpertOverlayPreparationPlan::build(*runtime, input.loader));
        }
        result.residency_ = std::make_shared<MoEOverlayParticipantResidencyRegistry>(MoEOverlayParticipantResidencyRegistry::Config{
            .owner_map = snapshot->owner_map, .local_participant_ids = std::move(local_ids),
            .num_layers = metadata.num_layers, .num_experts = metadata.num_experts, .initial_epoch = snapshot->epoch,
            .retained_epoch_capacity = 2, .collect_economy_service_measurements = dynamic,
            .projection_preparation = std::move(preparation), .first_model_layer = metadata.first_model_layer});
        PerfStatsCollector::addCounter("moe_overlay_residency", "production_authorities_created", 1.0, "model_setup", {},
            {{"dynamic", dynamic ? "true" : "false"}, {"maintenance_mode", moeRebalanceRuntimeModeToString(rebalance.mode)},
             {"participants", std::to_string(owner_map.participants().size())},
             {"local_participants", std::to_string(result.residency_->localParticipantIds().size())},
             {"tiers", std::to_string(plan.routed_tiers.size())}, {"world_rank", std::to_string(input.world_rank)},
             {"world_size", std::to_string(input.world_size)}, {"first_model_layer", std::to_string(metadata.first_model_layer)}});
        return result;
    }
}
