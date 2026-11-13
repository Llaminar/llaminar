/**
 * @file MoEOverlayPipelineStageBinding.cpp
 * @brief Authenticate stage identity before a PP child receives expert owners.
 *
 * All inspection occurs before graph construction. It reads immutable setup
 * metadata and the initial authority snapshot, never device execution state or
 * model/cache payloads. The same endpoint geometry used by residency is the
 * handoff authority; PP does not maintain a second ownership table.
 *
 * Local and global pipeline stages share the original rank-owned prepared
 * source and PMA handoff. Global topology remains immutable metadata; remote
 * execution, activation transport and model-context reuse are not impersonated.
 */
#include "MoEOverlayPipelineStageBinding.h"
#include "execution/factory/InferenceRunnerFactory.h"
#include "execution/local_execution/orchestrators/IRankOrchestrator.h"
#include "MoEOverlayCPUHostMaintenance.h"
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
        const auto &initial_owners = snapshot->originalOwnerMap();
        if (snapshot->ownershipKind() == MoEOverlayResidencyOwnershipKind::CPUJointFragments)
        {
            // Joint banks borrow the same source-owned fragment map. The scalar
            // snapshot arm is deliberately empty and cannot certify these banks.
            if (&prepared_owners != &initial_owners ||
                snapshot->jointOwnerMap().get() != &initial_owners)
                throw std::invalid_argument("Pipeline expert binding requires its original joint fragment banks");
        }
        else if (prepared_owners.owners() != initial_owners.owners() ||
                 prepared_owners.participants() != initial_owners.participants())
            throw std::invalid_argument("Pipeline expert binding prepared banks disagree with initial expert ownership");
        initial_owners.requireLayerGeometry(
            metadata.num_layers, metadata.num_experts, metadata.first_model_layer);
        const auto &plan = *snapshot->placement_plan;
        if (!plan.usesExpertOverlayAuthority() || plan.first_model_layer != stage.first_layer ||
            plan.authority_execution != resolveMoEOverlayAuthorityExecutionKind(plan))
            throw std::invalid_argument("Pipeline expert binding has a foreign placement interval");
        const auto domain = std::find_if(plan.domains.begin(), plan.domains.end(),
            [&](const auto &candidate) { return candidate.name == plan.effectiveBaseModelDomain(); });
        const bool global_cpu=domain!=plan.domains.end() && domain->scope==ExecutionDomainScope::NODE_LOCAL &&
            domain->participants.size()>1 &&
            std::all_of(domain->participants.begin(),domain->participants.end(),[](const auto& device){return device.isCPU();});
        if (domain == plan.domains.end() || domain->participants != participants ||
            domain->routed_compute_policy == RoutedExpertComputePolicy::Automatic ||
            (domain->scope != ExecutionDomainScope::RANK_LOCAL &&
             domain->scope != ExecutionDomainScope::SINGLE && !global_cpu))
            throw std::invalid_argument("Pipeline expert binding differs from its continuation participants");

        // Validate every local endpoint, including routed-only participants.
        // A matching vector length cannot prove device or layer ownership.
        const int local_rank = owners.mpi ? owners.mpi->rank() : domain->owner_rank;
        if (local_rank < 0 || (!global_cpu && domain->owner_rank != local_rank) ||
            (global_cpu && (!owners.mpi || owners.mpi->world_size()<=1 ||
                domain->world_ranks.size()!=participants.size() ||
                std::find(domain->world_ranks.begin(),domain->world_ranks.end(),local_rank)==domain->world_ranks.end())))
            throw std::invalid_argument("Pipeline expert binding requires its original executing MPI rank");
        std::vector<int> expected_local_ids;
        for (const auto &participant : initial_owners.participants())
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

        if (owners.host_maintenance)
            owners.host_maintenance->requireScope(owners.authority.get(), owners.residency.get(), metadata);

        return std::shared_ptr<const MoEOverlayPipelineStageBinding>(
            new MoEOverlayPipelineStageBinding(stage, std::move(metadata),
                std::move(participants), std::move(owners), snapshot->placement_plan));
    }

    std::shared_ptr<const MoEOverlayPipelineStageBinding>
    MoEOverlayPipelineStageBinding::withOriginalCPURankTransport(
        std::shared_ptr<MoEOverlayRankBatchTransportRegistry> registry) const
    {
        if(!registry || owners_.rank_transport || !owners_.mpi ||
            !std::all_of(participants_.begin(),participants_.end(),[](const auto& p){return p.isCPU();}))
            throw std::invalid_argument("CPU pipeline rank transport requires its original fresh stage binding");
        auto owners=owners_;owners.rank_transport=std::move(registry);
        return seal(stage_,metadata_,participants_,std::move(owners));
    }

    void MoEOverlayPipelineStageBinding::installRuntime(InferenceRunnerConfig &destination,
        std::span<const GlobalDeviceAddress> participants,
        std::span<const float> weights, CollectiveBackendType backend) const
    {
        if (!destination.pp_stage_config || !destination.cpu_serial_expert_storage.empty() || destination.moe_routed_expert_plan ||
            destination.moe_expert_overlay_residency_authority || destination.moe_expert_overlay_participant_residency ||
            destination.moe_expert_overlay_decode_histogram || destination.moe_expert_overlay_mpi_ctx ||
            destination.moe_rank_batch_transport_registry || destination.moe_device_controller_fabric ||
            destination.moe_expert_overlay_cpu_joint_context || destination.moe_overlay_cpu_preparation ||
            destination.moe_node_local_route_exchange || !destination.moe_node_local_route_transport_policy.unresolved())
            throw std::invalid_argument("Pipeline stage factory requires its exact scope and no preexisting expert owner");
        requireDestination(*destination.pp_stage_config, participants, weights, backend, destination.moe_rebalance.mode);
        auto placement = std::make_shared<MoERoutedExpertPlacementPlan>(*plan_);
        const auto domain = std::find_if(placement->domains.begin(), placement->domains.end(),
            [&](const auto &value) { return value.name == placement->effectiveBaseModelDomain(); });
        if (domain == placement->domains.end())
            throw std::logic_error("Sealed pipeline stage lost its original continuation domain");
        destination.moe_routed_expert_plan = std::move(placement);
        destination.cpu_serial_expert_storage = owners_.cpu_serial_expert_storage;
        destination.moe_expert_overlay_residency_authority = owners_.authority;
        destination.moe_expert_overlay_participant_residency = owners_.residency;
        destination.moe_expert_overlay_decode_histogram = owners_.histogram;
        destination.moe_expert_overlay_mpi_ctx = owners_.mpi;
        destination.moe_rank_batch_transport_registry = owners_.rank_transport;
        destination.moe_device_controller_fabric = owners_.device_fabric;
        destination.routed_expert_compute_policy = domain->routed_compute_policy;
    }

    void MoEOverlayPipelineStageBinding::requireRuntimeOwner(const IInferenceRunner &child,
        IInferenceRunner::MoEPipelineOwnershipValidation frontier) const
    {
        using Validation = IInferenceRunner::MoEPipelineOwnershipValidation;
        if (frontier != Validation::OwnersBound && frontier != Validation::ServingMaterialized)
            throw std::invalid_argument("Pipeline expert binding has an unknown setup frontier");
        if (const auto *rank_runner = dynamic_cast<const IRankOrchestrator *>(&child))
        {
            if (rank_runner->device_count() != static_cast<int>(participants_.size()))
                throw std::logic_error("MoE pipeline child lost an admitted TP participant");
            for (int member = 0; member < rank_runner->device_count(); ++member)
                if (!rank_runner->deviceRunner(member) || rank_runner->deviceRunner(member)->primaryDeviceId() !=
                    participants_[static_cast<std::size_t>(member)].toLocalDeviceId())
                    throw std::logic_error("MoE pipeline child has foreign ordered TP membership");
        }
        else {
            const auto domain=std::find_if(plan_->domains.begin(),plan_->domains.end(),
                [&](const auto& value){return value.name==plan_->effectiveBaseModelDomain();});
            const bool global_cpu=domain!=plan_->domains.end() && domain->scope==ExecutionDomainScope::NODE_LOCAL &&
                owners_.mpi && std::all_of(participants_.begin(),participants_.end(),[](const auto& p){return p.isCPU();});
            if(global_cpu) {
                const auto rank=std::find(domain->world_ranks.begin(),domain->world_ranks.end(),owners_.mpi->rank());
                if(rank==domain->world_ranks.end() || domain->world_ranks.size()!=participants_.size() ||
                    child.primaryDeviceId()!=participants_[rank-domain->world_ranks.begin()].toLocalDeviceId())
                    throw std::logic_error("CPU pipeline child has a foreign original rank endpoint");
            }
            else if(participants_.size()!=1u || child.primaryDeviceId()!=participants_.front().toLocalDeviceId())
                throw std::logic_error("MoE pipeline child has a foreign single-participant identity");
        }
        if (child.moeOverlayAuthorityExecution() != this->plan()->authority_execution)
            throw std::logic_error("MoE pipeline child authority differs from its own admitted owner");
        const auto &no_peer = this->owners().authority->noPeerResidency();
        if (no_peer)
        {
            const auto &metadata = this->metadata();
            no_peer->requireDestination(child.primaryDeviceId(), metadata.first_model_layer,
                metadata.num_layers, metadata.num_experts);
            if (no_peer->authorityExecution() != this->plan()->authority_execution)
                throw std::logic_error("MoE pipeline no-peer proof has a foreign authority locus");
        }
        else if (owners_.authority->maintenanceMode() == MoERebalanceRuntimeMode::Dynamic)
        {
            // A genuinely movable host stage needs its own admitted host
            // maintenance composition. A no-peer identity cannot certify it.
            if (this->plan()->authority_execution == MoEOverlayAuthorityExecutionKind::HostResident)
            {
                const auto &owner = this->owners().host_maintenance;
                if (!owner) throw std::logic_error("Movable host MoE pipeline stage has no admitted maintenance owner");
                owner->requireScope(this->owners().authority.get(), this->owners().residency.get(), this->metadata());
                if (frontier == Validation::ServingMaterialized && !owner->ready())
                    throw std::logic_error("Movable host MoE pipeline stage has no prepared maintenance family");
            }
            else if (frontier == Validation::ServingMaterialized &&
                !child.deviceResidentMoEOverlayMaintenanceReady())
                throw std::logic_error("Movable MoE pipeline stage has no complete native maintenance family");
        }
        if (frontier == Validation::ServingMaterialized &&
            !this->owners().residency->allInitialBanksReady())
            throw std::logic_error("MoE pipeline stage has incomplete physical prepared residency");
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
            stage.generation_domain != stage_.generation_domain ||
            !std::equal(participants.begin(), participants.end(), participants_.begin(), participants_.end()) ||
            !std::equal(weights.begin(), weights.end(), domain->weights.begin(), domain->weights.end()) ||
            backend != domain->backend ||
            mode != owners_.authority->maintenanceMode())
            throw std::invalid_argument("Pipeline expert runtime belongs to another stage, TP declaration or maintenance mode");
    }
}
