/**
 * @file MoEOverlayPipelineStageBinding.cpp
 * @brief Authenticate stage identity before a PP child receives expert owners.
 *
 * All inspection occurs before graph construction. It reads immutable setup
 * metadata and the initial authority snapshot, never device execution state or
 * model/cache payloads. The same endpoint geometry used by residency is the
 * handoff authority; PP does not maintain a second ownership table.
 */
#include "MoEOverlayPipelineStageBinding.h"
#include "MoEOverlayResidencyAuthority.h"
#include "MoEOverlayParticipantResidency.h"
#include "MoEOverlayNodeLocalDeviceControllerFabric.h"
#include "interfaces/IMPIContext.h"
#include <algorithm>
#include <limits>
#include <stdexcept>
#include <utility>

namespace llaminar2
{
    MoEOverlayPipelineStageBinding::MoEOverlayPipelineStageBinding(
        FactoryPPStageConfig stage, MoERoutedExpertModelMetadata metadata,
        std::vector<GlobalDeviceAddress> participants, Owners owners,
        std::shared_ptr<const MoERoutedExpertPlacementPlan> plan)
        : stage_(stage), metadata_(std::move(metadata)),
          participants_(std::move(participants)), owners_(std::move(owners)),
          plan_(std::move(plan)) {}

    std::shared_ptr<const MoEOverlayPipelineStageBinding>
    MoEOverlayPipelineStageBinding::seal(
        FactoryPPStageConfig stage, MoERoutedExpertModelMetadata metadata,
        std::vector<GlobalDeviceAddress> participants, Owners owners)
    {
        if (!stage.isValid() || participants.empty() || !owners.authority || !owners.residency ||
            metadata.num_layers <= 0 || metadata.num_experts <= 0 ||
            metadata.first_model_layer != stage.first_layer ||
            metadata.num_layers > std::numeric_limits<int>::max() - metadata.first_model_layer ||
            metadata.main_inference_layer_count != stage.last_layer ||
            (stage.has_embedding && stage.first_layer != 0) ||
            metadata.num_layers < stage.layerCount() ||
            (!stage.has_lm_head && metadata.num_layers != stage.layerCount()))
            throw std::invalid_argument("Pipeline expert binding requires exact stage and terminal MTP geometry");

        const auto snapshot = owners.authority->snapshot();
        if (!snapshot || !snapshot->valid() ||
            owners.residency->initialEpoch() != snapshot->epoch ||
            !owners.authority->retainsHistogram(owners.histogram.get()) ||
            (owners.authority->maintenanceMode() != MoERebalanceRuntimeMode::Off && !owners.histogram))
            throw std::invalid_argument("Pipeline expert binding requires one initial residency and histogram authority");
        const auto &prepared_owners = owners.residency->initialOwnerMap();
        if (prepared_owners.owners() != snapshot->owner_map.owners() ||
            prepared_owners.participants() != snapshot->owner_map.participants())
            throw std::invalid_argument("Pipeline expert binding prepared banks disagree with initial expert ownership");
        snapshot->owner_map.requireLayerGeometry(
            metadata.num_layers, metadata.num_experts, metadata.first_model_layer);
        const auto &plan = *snapshot->placement_plan;
        if (!plan.usesExpertOverlayAuthority() || plan.first_model_layer != stage.first_layer ||
            plan.authority_execution != resolveMoEOverlayAuthorityExecutionKind(plan))
            throw std::invalid_argument("Pipeline expert binding has a foreign placement interval");
        const auto domain = std::find_if(plan.domains.begin(), plan.domains.end(),
            [&](const auto &candidate) { return candidate.name == plan.effectiveBaseModelDomain(); });
        if (domain == plan.domains.end() || domain->participants != participants ||
            domain->routed_compute_policy == RoutedExpertComputePolicy::Automatic ||
            (domain->scope != ExecutionDomainScope::RANK_LOCAL &&
             domain->scope != ExecutionDomainScope::SINGLE))
            throw std::invalid_argument("Pipeline expert binding differs from its continuation participants");

        // Validate every local endpoint, including routed-only participants.
        // A matching vector length cannot prove device or layer ownership.
        const int local_rank = owners.mpi ? owners.mpi->rank() : domain->owner_rank;
        if (local_rank < 0 || domain->owner_rank != local_rank)
            throw std::invalid_argument("Pipeline expert binding requires a resolved local rank");
        std::vector<int> expected_local_ids;
        for (const auto &participant : snapshot->owner_map.participants())
        {
            if (!participant.world_rank_known)
                throw std::invalid_argument("Pipeline expert binding has an unresolved participant rank");
            if (participant.world_rank != local_rank) continue;
            expected_local_ids.push_back(participant.participant_id);
            const auto endpoint = owners.residency->endpoint(participant.participant_id);
            if (!endpoint || endpoint->device() != participant.device ||
                endpoint->firstModelLayer() != metadata.first_model_layer ||
                endpoint->numLayers() != metadata.num_layers ||
                endpoint->numExperts() != metadata.num_experts)
                throw std::invalid_argument("Pipeline expert binding has a foreign prepared-bank endpoint");
        }
        if (expected_local_ids.empty() || owners.residency->localParticipantIds() != expected_local_ids)
            throw std::invalid_argument("Pipeline expert binding requires its complete local participant set");
        if (owners.device_fabric)
        {
            const auto &layout = owners.device_fabric->layout();
            if (!layout.valid() || layout.header.first_model_layer != metadata.first_model_layer ||
                layout.header.num_layers != static_cast<std::uint32_t>(metadata.num_layers) ||
                layout.header.num_experts != static_cast<std::uint32_t>(metadata.num_experts) ||
                owners.device_fabric->localParticipantIds() != expected_local_ids)
                throw std::invalid_argument("Pipeline expert binding has a foreign device-controller fabric");
        }

        return std::shared_ptr<const MoEOverlayPipelineStageBinding>(
            new MoEOverlayPipelineStageBinding(stage, std::move(metadata),
                std::move(participants), std::move(owners), snapshot->placement_plan));
    }

    void MoEOverlayPipelineStageBinding::requireDestination(
        const FactoryPPStageConfig &stage,
        std::span<const GlobalDeviceAddress> participants,
        std::span<const float> weights, CollectiveBackendType backend,
        MoERebalanceRuntimeMode mode) const
    {
        const auto domain = std::find_if(plan_->domains.begin(), plan_->domains.end(),
            [&](const auto &candidate) { return candidate.name == plan_->effectiveBaseModelDomain(); });
        if (stage.first_layer != stage_.first_layer || stage.last_layer != stage_.last_layer ||
            stage.has_embedding != stage_.has_embedding || stage.has_lm_head != stage_.has_lm_head ||
            !std::equal(participants.begin(), participants.end(), participants_.begin(), participants_.end()) ||
            !std::equal(weights.begin(), weights.end(), domain->weights.begin(), domain->weights.end()) ||
            backend != domain->backend ||
            mode != owners_.authority->maintenanceMode())
            throw std::invalid_argument("Pipeline expert runtime belongs to another stage, TP declaration or maintenance mode");
    }
}
