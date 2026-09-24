/**
 * @file DeviceGraphOrchestratorMovement.cpp
 * @brief Terminal native movement projection and passive runner diagnostics.
 *
 * The captured finalizer authors the journal. This translation unit performs
 * no device I/O: it validates completed bytes against frozen topology and
 * exposes immutable receipts through the public runner. Optional PerfStats
 * mirrors share the existing durable movement vocabulary; no second protocol
 * or host placement authority is introduced for homogeneous GPU domains.
 */
#include "DeviceGraphOrchestrator.h"
#include "execution/compute_stages/stages/MoEDeviceRebalanceStage.h"
#include "execution/moe/MoERoutedExpertPlacementPlan.h"
#include "execution/moe/NativeMoEMovementRequestIdentity.h"
#include "utils/PerfStatsCollector.h"

namespace llaminar2
{
    const MoEDeviceRebalanceStage *DeviceGraphOrchestrator::nativeMoEMovementOwnerStage() const
    {
        if (!usesParticipantLocalDeviceMoERebalanceController() || usesTopologyWideDeviceMoEOverlayController())
            return nullptr;
        const auto &graph = device_moe_rebalance_maintenance_graph_.graph;
        if (!graph)
            return nullptr;
        const MoEDeviceRebalanceStage *owner = nullptr;
        for (const auto &name : graph->getExecutionOrder())
        {
            const auto *node = graph->getNode(name);
            const auto *stage = node && node->stage
                ? dynamic_cast<const MoEDeviceRebalanceStage *>(node->stage.get()) : nullptr;
            if (!stage || stage->getParams().phase != DeviceMoERebalanceStagePhase::PlanCopyApply)
                continue;
            const auto &config = stage->getParams().config;
            if (config.participant_id != config.root_participant)
                continue;
            if (owner)
                throw std::logic_error("Native MoE movement has multiple root publication stages");
            owner = stage;
        }
        return owner;
    }

    MoEOptimizationStatus DeviceGraphOrchestrator::moeOptimizationStatus() const
    {
        if (!nativeMoEMovementOwnerStage() && !native_moe_movement_archive_)
            return {};
        return {.authority = MoEOptimizationAuthority::Device,
            .state = MoEOptimizationLifecycleState::Active,
            .activity = MoEOptimizationActivityState::DeviceOwned,
            .published_movement_waves = native_moe_movement_archive_ ? native_moe_movement_archive_->totals().transactions : 0,
            .completed_movement = native_moe_movement_archive_ ? native_moe_movement_archive_->totals() : MoEOptimizationMovementTotals{}};
    }

    MoEOptimizationMovementLedger DeviceGraphOrchestrator::moeOptimizationMovementLedger() const
    {
        return native_moe_movement_archive_ ? native_moe_movement_archive_->ledger() : MoEOptimizationMovementLedger{};
    }

    void DeviceGraphOrchestrator::archiveCompletedNativeMoEMovement(const MoEDeviceRebalanceStage &stage,
        const DeviceMoERebalanceGraphControllerState &controller, uint64_t generation,
        std::span<const DeviceMoERebalanceMovementWave> waves,
        std::span<const DeviceMoERebalanceMovementEdge> edges)
    {
        const auto &params = stage.getParams();
        const auto &config = params.config;
        if (&stage != nativeMoEMovementOwnerStage() || !graph_builder_ ||
            !graph_builder_->config().moe.routed_expert_plan)
            throw std::invalid_argument("Native MoE movement archive has no authenticated root/request binding");
        const auto request_epoch = nativeMoEMovementArchiveRequestEpoch(controller.dispatch_ticket,
            generation, config.participant_id, config.participant_count);
        if (!native_moe_movement_archive_)
        {
            const auto &plan = *graph_builder_->config().moe.routed_expert_plan;
            if (plan.routed_tiers.size() != 1)
                throw std::invalid_argument("Native MoE movement requires exactly one frozen routed tier");
            const auto &tier = plan.routed_tiers.front();
            const auto domain = std::find_if(plan.domains.begin(), plan.domains.end(),
                [&](const auto &value) { return value.name == tier.domain; });
            if (domain == plan.domains.end() || domain->participants.size() != config.participant_count ||
                config.root_participant >= config.participant_count ||
                domain->participants[config.root_participant].toLocalDeviceId() != state_.device_id ||
                (!domain->world_ranks.empty() && domain->world_ranks.size() != domain->participants.size()))
                throw std::invalid_argument("Native MoE root/domain participant geometry disagrees with captured policy");
            const auto view = stage.movementJournalView();
            NativeMoEMovementArchiveConfig geometry{.workspace_generation = generation,
                .layers = config.num_layers, .experts = config.num_experts,
                .wave_capacity = view.wave_capacity, .edge_capacity = view.edge_capacity};
            for (std::size_t i = 0; i < domain->participants.size(); ++i)
                geometry.participants.push_back({domain->participants[i].toLocalDeviceId(),
                    domain->world_ranks.empty() ? domain->owner_rank : domain->world_ranks[i], tier.priority});
            native_moe_movement_archive_.emplace(std::move(geometry));
        }
        auto &archive = *native_moe_movement_archive_;
        const auto first_economy = archive.ledger().economy.size();
        const auto first_edge = archive.observe(request_epoch, generation,
            controller.movement_journal, waves, edges);

        // Mirror only newly archived completed records. Repeated terminal
        // observations cannot double count; disabling telemetry changes no
        // authoritative state and cannot make missing device evidence appear.
        if (!PerfStatsCollector::isDomainEnabled("moe_overlay_controller"))
            return;
        const auto &ledger = archive.ledger();
        for (std::size_t i = first_economy; i < ledger.economy.size(); ++i)
        {
            const auto &economy = ledger.economy[i];
            const auto wave = std::find_if(waves.begin(), waves.end(),
                [&](const auto &value) { return value.candidate_epoch == economy.candidate_epoch; });
            if (wave == waves.end())
                throw std::logic_error("Archived native movement lost its terminal payload receipt");
            const std::map<std::string, std::string> tags{
                {"transaction", std::to_string(economy.transaction)},
                {"candidate_epoch", std::to_string(economy.candidate_epoch)},
                {"policy_owner", "device"}, {"policy", "native_load_spread"}};
            PerfStatsCollector::addCounter("moe_overlay_controller", "dynamic_movement_transactions", 1.0,
                "maintenance", state_.device_id.toString(), tags);
            PerfStatsCollector::addCounter("moe_overlay_controller", "dynamic_physical_bytes",
                static_cast<double>(wave->physical_payload_bytes), "maintenance", state_.device_id.toString(), tags);
        }
        for (std::size_t i = first_edge; i < ledger.edges.size(); ++i)
        {
            const auto &edge = ledger.edges[i];
            PerfStatsCollector::addCounter("moe_overlay_controller", "dynamic_migration_edges", 1.0,
                "maintenance", state_.device_id.toString(),
                {{"transaction", std::to_string(edge.transaction)}, {"candidate_epoch", std::to_string(edge.candidate_epoch)},
                 {"layer", std::to_string(edge.layer)}, {"expert", std::to_string(edge.expert)},
                 {"source_participant", std::to_string(edge.source_participant)},
                 {"destination_participant", std::to_string(edge.destination_participant)},
                 {"source_priority", std::to_string(edge.source_priority)},
                 {"destination_priority", std::to_string(edge.destination_priority)},
                 {"source_world_rank", std::to_string(edge.source_world_rank)},
                 {"destination_world_rank", std::to_string(edge.destination_world_rank)},
                 {"source_device", edge.source_device.toString()}, {"destination_device", edge.destination_device.toString()},
                 {"movement_axis", "participant_placement"}, {"direction", "same_priority"},
                 {"blocking_inference", "false"}, {"policy_owner", "device"}, {"policy", "native_load_spread"}});
        }
    }
}
