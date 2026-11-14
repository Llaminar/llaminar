/**
 * @file MoEOverlayCapacityAdmission.cpp
 * @brief Hardware-bound ExpertOverlay capacity admission implementation.
 *
 * Logical participants are joined to physical memory through typed rank/device
 * keys before the pure byte resolver runs.  All policy decisions come from
 * scoped enums and integer priorities. Opaque tier and domain labels are used
 * solely for identity and diagnostics. A pipeline's bounded cache search uses
 * complete joint candidates, preserving one CPU/GPU admission certificate for
 * all stages even when one participant limits its local replica grant.
 */

#include "MoEOverlayCapacityAdmission.h"
#include "utils/Assertions.h"
#include "MoEOverlayNoPeerResidency.h"
#include "MoEOverlayPhysicalLaneDemand.h"
#include "execution/mpi_orchestration/DeviceInventory.h"

#include <algorithm>
#include <limits>
#include <map>
#include <set>
#include <sstream>
#include <stdexcept>
#include <utility>

namespace llaminar2
{
    namespace
    {
        struct PhysicalKey
        {
            int world_rank = -1;
            DeviceId device = DeviceId::invalid();

            bool operator<(const PhysicalKey &other) const noexcept
            {
                if (world_rank != other.world_rank)
                    return world_rank < other.world_rank;
                return device < other.device;
            }
        };

        /** @return Observed rank inventory for one world rank, or null. */
        [[nodiscard]] const RankInventory *rankByObservedRank(
            const ClusterInventory &inventory, int rank)
        {
            for (const auto &candidate : inventory.ranks)
                if (candidate.rank == rank)
                    return &candidate;
            return nullptr;
        }

        /**
         * @brief Resolve the overlay-uniform CPU prepared layout from live inventory.
         *
         * This is the SAME observation and refusal the hardware binder publishes:
         * every CPU participant's owning rank is observed and paired; mixed
         * layouts and unobserved owner ranks are typed rejections. Absent when
         * the overlay declares no CPU participant. A plan that carries the
         * binder's own observation never reaches this derivation.
         */
        [[nodiscard]] std::optional<
            cpu::native_quantised::CPUNativeQuantisedLayout>
        resolveUniformCpuPreparedLayoutFromInventory(
            const ClusterInventory &inventory,
            const MoERoutedExpertPlacementPlan &plan)
        {
            std::vector<std::pair<std::string, int>> cpu_endpoint_owners;
            for (const auto &domain : plan.domains)
            {
                if (!domain.world_ranks.empty() &&
                    domain.world_ranks.size() != domain.participants.size())
                {
                    throw std::invalid_argument(
                        "ExpertOverlay capacity admission cannot resolve CPU "
                        "endpoint ranks for domain '" +
                        domain.name +
                        "': the domain declares a different rank count than "
                        "participants");
                }
                for (std::size_t index = 0;
                     index < domain.participants.size(); ++index)
                {
                    if (domain.participants[index].device_type !=
                        DeviceType::CPU)
                        continue;
                    const int rank = !domain.world_ranks.empty()
                        ? domain.world_ranks[index]
                        : domain.owner_rank;
                    if (rank < 0)
                    {
                        throw std::invalid_argument(
                            "ExpertOverlay CPU endpoint in domain '" +
                            domain.name +
                            "' has no bound rank to observe; declare "
                            "world_ranks or owner_rank");
                    }
                    cpu_endpoint_owners.push_back({domain.name, rank});
                }
            }
            if (cpu_endpoint_owners.empty())
                return std::nullopt;
            const RankInventory *first = rankByObservedRank(
                inventory, cpu_endpoint_owners.front().second);
            if (!first)
            {
                throw std::invalid_argument(
                    "MoE overlay CPU endpoint references an unobserved owner rank");
            }
            const auto uniform = first->cpu_prepared_layout;
            for (const auto &endpoint : cpu_endpoint_owners)
            {
                const RankInventory *owner =
                    rankByObservedRank(inventory, endpoint.second);
                if (!owner)
                {
                    throw std::invalid_argument(
                        "MoE overlay CPU endpoint references an unobserved owner rank");
                }
                if (owner->cpu_prepared_layout != uniform)
                {
                    throw std::invalid_argument(
                        "MoE overlay domain '" + endpoint.first +
                        "' pairs CPU rank " +
                        std::to_string(endpoint.second) +
                        " executing prepared layout '" +
                        cpu::native_quantised::preparedLayoutName(
                            owner->cpu_prepared_layout) +
                        "' with CPU rank " +
                        std::to_string(
                            cpu_endpoint_owners.front().second) +
                        " executing '" +
                        cpu::native_quantised::preparedLayoutName(
                            uniform) +
                        "'; cross-layout CPU endpoint pairing is refused at "
                        "planning time before any allocation or prepared-weight "
                        "transfer");
                }
            }
            return uniform;
        }

        /** @return Whether the plan declares any CPU routed participant. */
        [[nodiscard]] bool planDeclaresCpuParticipant(
            const MoERoutedExpertPlacementPlan &plan)
        {
            for (const auto &domain : plan.domains)
                for (const auto &participant : domain.participants)
                    if (participant.device_type == DeviceType::CPU)
                        return true;
            return false;
        }

        [[nodiscard]] const RoutedExpertDomain &requireDomain(
            const MoERoutedExpertPlacementPlan &plan,
            const RoutedExpertTier &tier)
        {
            const auto found = std::find_if(
                plan.domains.begin(), plan.domains.end(),
                [&](const auto &domain)
                { return domain.name == tier.domain; });
            if (found == plan.domains.end())
            {
                throw std::invalid_argument(
                    "ExpertOverlay capacity tier '" + tier.name +
                    "' references unknown domain '" + tier.domain + "'");
            }
            return *found;
        }

        [[nodiscard]] int boundParticipantWorldRank(
            const RoutedExpertDomain &domain,
            std::size_t participant_index)
        {
            if (participant_index < domain.world_ranks.size())
                return domain.world_ranks[participant_index];
            if (domain.owner_rank >= 0)
                return domain.owner_rank;

            throw std::invalid_argument(
                "ExpertOverlay capacity domain '" + domain.name +
                "' has a participant without a hardware-bound world rank");
        }

        [[nodiscard]] MoEOverlayTierCopyPolicy copyPolicy(
            const RoutedExpertDomain &domain)
        {
            switch (domain.routed_compute_policy)
            {
            case RoutedExpertComputePolicy::Apportioned:
                return MoEOverlayTierCopyPolicy::Apportioned;
            case RoutedExpertComputePolicy::GateUpOwnedDownColumns:
                return MoEOverlayTierCopyPolicy::GateUpOwnedDownColumns;
            case RoutedExpertComputePolicy::Replicated:
                return MoEOverlayTierCopyPolicy::Replicated;
            case RoutedExpertComputePolicy::TensorSharded:
                // A degree-one CPU tensor policy owns the complete matrix.
                // Only multi-member CPU domains require fragment cohorts;
                // the authored public compute policy remains unchanged.
                if (domain.participants.size() == 1 && domain.participants.front().isCPU())
                    return MoEOverlayTierCopyPolicy::Apportioned;
                return MoEOverlayTierCopyPolicy::TensorSharded;
            case RoutedExpertComputePolicy::Unspecified:
            case RoutedExpertComputePolicy::Automatic:
                throw std::invalid_argument(
                    "ExpertOverlay capacity domain '" + domain.name +
                    "' has an unresolved routed compute policy");
            }
            throw std::logic_error(
                "Unknown routed-expert compute policy during capacity admission");
        }

        [[nodiscard]] std::size_t checkedAdd(
            std::size_t left,
            std::size_t right,
            const char *what)
        {
            if (right > std::numeric_limits<std::size_t>::max() - left)
            {
                throw std::overflow_error(
                    std::string("ExpertOverlay capacity ") + what +
                    " overflows size_t");
            }
            return left + right;
        }

        [[nodiscard]] std::size_t checkedMultiply(
            std::size_t left,
            std::size_t right,
            const char *what)
        {
            if (left != 0 &&
                right > std::numeric_limits<std::size_t>::max() / left)
            {
                throw std::overflow_error(
                    std::string("ExpertOverlay capacity ") + what +
                    " overflows size_t");
            }
            return left * right;
        }

        /**
         * @brief Convert the legacy tier-local byte cap into an exact layer cap.
         *
         * The scalar is retained temporarily as an additional upper bound while
         * the CLI moves to physical participant budgets. It never substitutes
         * for physical admission. Apportioned tiers are charged in their actual
         * round-robin participant order; replicated tiers charge every copy.
         */
        [[nodiscard]] int exactLegacyLayerCap(
            std::size_t byte_cap,
            int num_experts,
            std::size_t layer_index,
            MoEOverlayTierCopyPolicy policy,
            const std::vector<DeviceId> &participant_devices,
            const std::vector<MoEOverlayPreparedExpertFootprint> &footprints)
        {
            if (byte_cap == 0)
                return num_experts;

            std::size_t used = 0;
            int admitted = 0;
            for (; admitted < num_experts; ++admitted)
            {
                std::size_t next = 0;
                if (policy != MoEOverlayTierCopyPolicy::Replicated)
                {
                    const auto participant =
                        static_cast<std::size_t>(admitted) %
                        participant_devices.size();
                    next = footprints[layer_index].liveBytes(
                        participant_devices[participant]);
                }
                else
                {
                    for (const auto device : participant_devices)
                    {
                        next = checkedAdd(
                            next,
                            footprints[layer_index].liveBytes(device),
                            "legacy replicated tier byte cap");
                    }
                }
                if (used > byte_cap || next > byte_cap - used)
                    break;
                used += next;
            }
            return admitted;
        }
    } // namespace

    MoEOverlayMigrationStorageKind
    MoEOverlayCapacityAdmission::selectMigrationStorage(
        const MoERoutedExpertPlacementPlan &plan,
        MoERebalanceRuntimeMode mode)
    {
        if (mode != MoERebalanceRuntimeMode::Dynamic)
            return MoEOverlayMigrationStorageKind::Disabled;
        if (!plan.usesExpertOverlayAuthority() ||
            plan.authority_execution ==
                MoEOverlayAuthorityExecutionKind::Unresolved)
        {
            throw std::invalid_argument(
                "Dynamic ExpertOverlay migration storage requires a normalized authority plan");
        }

        // The frozen full-residency proof is sealed after quota resolution.
        // Geometry already proves there is no second payload endpoint to price.
        if (MoEOverlayNoPeerResidency::hasNoPeerTopology(plan))
            return MoEOverlayMigrationStorageKind::NoPeer;

        if (plan.authority_execution !=
                MoEOverlayAuthorityExecutionKind::DeviceResident ||
            plan.routed_tiers.size() != 1u)
        {
            return MoEOverlayMigrationStorageKind::
                PhysicalResidencyFabric;
        }

        const auto &domain = requireDomain(
            plan, plan.routed_tiers.front());
        if (domain.participants.size() < 2u)
        {
            return MoEOverlayMigrationStorageKind::
                PhysicalResidencyFabric;
        }
        const DeviceType device_type =
            domain.participants.front().device_type;
        const bool homogeneous_gpu = std::all_of(
            domain.participants.begin(),
            domain.participants.end(),
            [device_type](const GlobalDeviceAddress &participant)
            {
                return participant.isGPU() &&
                       participant.device_type == device_type;
            });
        const bool native_collective =
            domain.backend == CollectiveBackendType::AUTO ||
            (device_type == DeviceType::CUDA &&
             domain.backend == CollectiveBackendType::NCCL) ||
            (device_type == DeviceType::ROCm &&
             domain.backend == CollectiveBackendType::RCCL);
        const auto owner_rank = domain.primaryWorldRank();
        if (!homogeneous_gpu || !native_collective || !owner_rank)
        {
            return MoEOverlayMigrationStorageKind::
                PhysicalResidencyFabric;
        }

        for (std::size_t participant = 0;
             participant < domain.participants.size();
             ++participant)
        {
            const int rank = participant < domain.world_ranks.size()
                                 ? domain.world_ranks[participant]
                                 : domain.owner_rank;
            if (rank != *owner_rank)
            {
                return MoEOverlayMigrationStorageKind::
                    PhysicalResidencyFabric;
            }
        }
        return MoEOverlayMigrationStorageKind::DeviceTransferDirectory;
    }

    std::vector<MoEOverlayBoundTierParticipant>
    MoEOverlayCapacityAdmission::boundParticipants(
        const MoERoutedExpertPlacementPlan &plan)
    {
        if (!plan.usesExpertOverlayAuthority())
        {
            throw std::invalid_argument(
                "ExpertOverlay participant enumeration requires an enabled tiered plan");
        }

        std::vector<MoEOverlayBoundTierParticipant> result;
        int participant_id = 0;
        for (std::size_t tier_index = 0;
             tier_index < plan.routed_tiers.size();
             ++tier_index)
        {
            const auto &tier = plan.routed_tiers[tier_index];
            const auto &domain = requireDomain(plan, tier);
            if (domain.participants.empty())
            {
                throw std::invalid_argument(
                    "ExpertOverlay capacity domain '" + domain.name +
                    "' has no participants");
            }
            for (std::size_t domain_index = 0;
                 domain_index < domain.participants.size();
                 ++domain_index)
            {
                const DeviceId device =
                    domain.participants[domain_index].toLocalDeviceId();
                if (!device.is_valid() ||
                    (!device.is_cpu() && !device.is_gpu()))
                {
                    throw std::invalid_argument(
                        "ExpertOverlay capacity domain '" + domain.name +
                        "' contains an unsupported participant device");
                }
                result.push_back({
                    .participant_id = participant_id++,
                    .tier_index = static_cast<int>(tier_index),
                    .domain_participant_index =
                        static_cast<int>(domain_index),
                    .world_rank = boundParticipantWorldRank(
                        domain, domain_index),
                    .device = device,
                });
            }
        }
        return result;
    }

    std::map<DeviceId,std::size_t> MoEOverlayCapacityAdmission::executionStreamCounts(
        const MoERoutedExpertPlacementPlan &plan,int world_rank,std::size_t maximum_execution_streams)
    {
        if(world_rank<0 || !maximum_execution_streams)
            throw std::invalid_argument("Movement execution streams require an original rank and positive width");
        std::map<DeviceId,std::size_t> result;
        for(const auto& member:boundParticipants(plan))if(member.world_rank==world_rank && member.device.is_gpu()) {
            auto& count=result[member.device];
            count=checkedAdd(count,maximum_execution_streams,"original shared execution stream population");
        }
        return result;
    }

    std::vector<MoEOverlayTransferStagingCharge>
    MoEOverlayCapacityAdmission::transferStagingBOM(
        const MoERoutedExpertPlacementPlan &plan,
        int world_size,
        const MoEOverlayCapacityAdmissionPolicy &policy,
        const std::vector<MoEOverlayLayerWeightManifest>& original_manifest,
        int original_experts)
    {
        if (!policy.usesPhysicalResidencyFabric())
            return {};
        if (world_size <= 0 || policy.staging_capacity_bytes == 0 ||
            policy.shadow_slots_per_endpoint_layer == 0 ||
            policy.maximum_concurrent_cycles == 0 ||
            policy.maximum_execution_streams == 0 ||
            policy.maximum_execution_streams >
                policy.maximum_concurrent_cycles ||
            policy.maximum_cycles_per_layer == 0 ||
            policy.shadow_slots_per_endpoint_layer <
                MoEOverlayCapacityAdmissionPolicy::
                    requiredShadowSlotsPerEndpointLayer(
                        policy.maximum_concurrent_cycles,
                        policy.maximum_cycles_per_layer) ||
            policy.distributed_transport != (world_size > 1))
        {
            throw std::invalid_argument(
                "ExpertOverlay migration fabric requires positive capacities and a communicator-consistent distributed policy");
        }

        const auto demand=configuredMoEOverlayLaneDemand(plan,original_manifest,original_experts,
            world_size,policy.maximum_concurrent_cycles,policy.maximum_cycles_per_layer);
        const auto counts=demand.materializedCounts();
        std::map<PhysicalKey,std::size_t> bytes_by_resource;
        const auto add=[&](int rank,DeviceId device,size_t count) {
            const auto bytes=checkedMultiply(count,policy.staging_capacity_bytes,"physical piece staging chunks");
            auto& total=bytes_by_resource[{rank,device}];total=checkedAdd(total,bytes,"physical piece staging BOM");
        };
        for(const auto& [key,count]:counts) {
            switch(key.kind) {
            case MoEOverlayPhysicalLaneKind::LocalDevice:
                if(key.source.is_cpu()!=key.destination.is_cpu()) {
                    add(key.world_rank,key.source.is_gpu()?key.source:key.destination,count);
                    add(key.world_rank,DeviceId::cpu(),count);
                } else if(key.source.is_gpu() && key.destination.is_gpu() && key.source.type!=key.destination.type)
                    add(key.world_rank,DeviceId::cpu(),checkedMultiply(count,4,"two runtime-pinned regions per double-buffered blob"));
                break;
            case MoEOverlayPhysicalLaneKind::RemoteSource:
            case MoEOverlayPhysicalLaneKind::RemoteDestination:
                if(policy.distributed_transport) {
                    add(key.world_rank,key.kind==MoEOverlayPhysicalLaneKind::RemoteSource?key.source:key.destination,count);
                    add(key.world_rank,DeviceId::cpu(),count);
                }
                break;
            case MoEOverlayPhysicalLaneKind::RemoteProjectionDirectory:
                if(policy.distributed_transport)for(int rank=0;rank<world_size;++rank)add(rank,DeviceId::cpu(),count);
                break;
            case MoEOverlayPhysicalLaneKind::LocalCPUWorker:
                // Existing NUMA workers copy admitted final roots directly;
                // neither a rectangle nor a compact tail adds CPU scratch.
                break;
            }
        }

        std::vector<MoEOverlayTransferStagingCharge> result;
        result.reserve(bytes_by_resource.size());
        for (const auto &[key, bytes] : bytes_by_resource)
        {
            if (bytes == 0)
                continue;
            result.push_back({
                .world_rank = key.world_rank,
                .device = key.device,
                .bytes = bytes,
            });
        }
        return result;
    }

    MoEOverlayCapacityResolverInput
    MoEOverlayCapacityAdmission::buildResolverInput(
        const MoERoutedExpertPlacementPlan &plan,
        int num_experts,
        const std::vector<MoEOverlayLayerWeightManifest> &layer_weight_manifest,
        const std::vector<MoEOverlayBoundPhysicalMemoryBudget> &physical_budgets,
        const MoEOverlayCapacityAdmissionPolicy &policy,
        const ClusterInventory *inventory)
    {
        if (!plan.usesExpertOverlayAuthority())
        {
            throw std::invalid_argument(
                "ExpertOverlay capacity admission requires an enabled tiered plan");
        }
        if (num_experts <= 0 || layer_weight_manifest.empty())
        {
            throw std::invalid_argument(
                "ExpertOverlay capacity admission requires positive model geometry");
        }
        if (plan.first_model_layer != layer_weight_manifest.front().layer_idx)
            throw std::invalid_argument(
                "ExpertOverlay capacity admission received another stage's layer manifest");
        if (policy.overlay_world_size <= 0 ||
            (policy.migration_storage !=
                 MoEOverlayMigrationStorageKind::Disabled &&
             policy.migration_storage !=
                 MoEOverlayMigrationStorageKind::DeviceTransferDirectory &&
             policy.migration_storage !=
                 MoEOverlayMigrationStorageKind::PhysicalResidencyFabric &&
             policy.migration_storage != MoEOverlayMigrationStorageKind::NoPeer))
        {
            throw std::invalid_argument(
                "ExpertOverlay capacity admission received an invalid migration storage policy");
        }
        if (policy.migration_storage == MoEOverlayMigrationStorageKind::NoPeer &&
            !MoEOverlayNoPeerResidency::hasNoPeerTopology(plan))
            throw std::invalid_argument("No-peer admission cannot exempt a movable topology");
        if (policy.usesPhysicalResidencyFabric() &&
            (policy.shadow_slots_per_endpoint_layer == 0 ||
             policy.staging_capacity_bytes == 0 ||
             policy.maximum_concurrent_cycles == 0 ||
             policy.maximum_execution_streams == 0 ||
             policy.maximum_execution_streams >
                 policy.maximum_concurrent_cycles ||
             policy.maximum_cycles_per_layer == 0 ||
             policy.device_transfer_directory_capacity.total_slots != 0u ||
             policy.device_rebalance_workspace_capacity.has_value() ||
             policy.shadow_slots_per_endpoint_layer <
                 MoEOverlayCapacityAdmissionPolicy::
                     requiredShadowSlotsPerEndpointLayer(
                         policy.maximum_concurrent_cycles,
                         policy.maximum_cycles_per_layer)))
        {
            throw std::invalid_argument(
                "ExpertOverlay migration admission requires staging and one endpoint/layer shadow slot per admitted same-layer cycle");
        }
        if (policy.usesDeviceTransferDirectory())
        {
            const auto &capacity =
                policy.device_transfer_directory_capacity;
            if (capacity.active_slots == 0u ||
                capacity.staging_slots == 0u ||
                capacity.total_slots == 0u ||
                static_cast<std::uint64_t>(capacity.active_slots) +
                        static_cast<std::uint64_t>(capacity.staging_slots) !=
                    capacity.total_slots ||
                !policy.device_rebalance_workspace_capacity.has_value() ||
                policy.shadow_slots_per_endpoint_layer != 0u ||
                policy.staging_capacity_bytes != 0u ||
                policy.distributed_transport)
            {
                throw std::invalid_argument(
                    "Device transfer-directory admission requires coherent buffered capacity and forbids physical-fabric storage");
            }
        }
        if (!policy.migrationEnabled() &&
            (policy.device_transfer_directory_capacity.total_slots != 0u ||
             policy.device_rebalance_workspace_capacity.has_value() ||
             policy.shadow_slots_per_endpoint_layer != 0u ||
             policy.staging_capacity_bytes != 0u ||
             policy.distributed_transport))
        {
            throw std::invalid_argument(
                "Movement-disabled ExpertOverlay admission cannot retain migration storage");
        }

        const auto footprints =
            MoEOverlayCapacityResolver::preparedFootprints(
                layer_weight_manifest);
        const std::size_t layer_count = footprints.size();

        const auto bound_participants = boundParticipants(plan);
        const auto staging_bom = transferStagingBOM(
            plan, policy.overlay_world_size, policy,layer_weight_manifest,num_experts);
        std::map<PhysicalKey, std::size_t> staging_by_key;
        std::map<PhysicalKey, std::size_t> workspace_by_key;
        for (const auto &charge : staging_bom)
        {
            staging_by_key.emplace(
                PhysicalKey{charge.world_rank, charge.device},
                charge.bytes);
        }
        if (policy.usesDeviceTransferDirectory())
        {
            if (plan.routed_tiers.size() != 1u ||
                bound_participants.size() < 2u)
            {
                throw std::invalid_argument(
                    "Device transfer-directory admission requires one multi-participant routed tier");
            }
            const int owner_rank = bound_participants.front().world_rank;
            const DeviceType backend_type =
                bound_participants.front().device.type;
            const auto profile =
                DeviceMoETransferSlotDirectory::
                    profileForLayerWeightManifest(
                        layer_weight_manifest,
                        moeOverlayMovableProjections(copyPolicy(
                            requireDomain(plan, plan.routed_tiers.front()))));
            const auto directory_bom =
                DeviceMoETransferSlotDirectory::allocationBOM(
                    policy.device_transfer_directory_capacity,
                    profile);
            const auto &workspace_capacity =
                *policy.device_rebalance_workspace_capacity;
            if (workspace_capacity.num_layers != layer_count ||
                workspace_capacity.num_experts !=
                    static_cast<std::uint32_t>(num_experts) ||
                workspace_capacity.participant_count !=
                    bound_participants.size() ||
                workspace_capacity.local_transfer_slot_count !=
                    policy.device_transfer_directory_capacity.total_slots ||
                workspace_capacity.phase !=
                    DeviceMoERebalanceStagePhase::PlanCopyApply ||
                workspace_capacity.transfer_mode !=
                    DeviceMoERebalanceTransferMode::CompactTransferSlots)
            {
                throw std::invalid_argument(
                    "Device transfer-directory workspace capacity disagrees with its admitted topology or directory");
            }
            const DeviceMoERebalanceWorkspaceBinding workspace_binding{
                .capacity = workspace_capacity,
                .collective_payload_slot_bytes =
                    DeviceMoERebalanceWorkspaceContract::
                        collectivePayloadSlotBytes(
                            profile.max_wire_payload_bytes),
                .workspace_suffix = "capacity_admission",
            };
            const std::size_t maintenance_workspace_bytes =
                DeviceMoERebalanceWorkspaceContract::allocationBytes(
                    workspace_binding);
            for (const auto &participant : bound_participants)
            {
                if (!participant.device.is_gpu() ||
                    participant.device.type != backend_type ||
                    participant.world_rank != owner_rank)
                {
                    throw std::invalid_argument(
                        "Device transfer-directory admission requires one rank-local homogeneous GPU domain");
                }
                auto &bytes = staging_by_key[PhysicalKey{
                    participant.world_rank, participant.device}];
                bytes = checkedAdd(
                    bytes,
                    directory_bom.total_bytes,
                    "device transfer-directory storage");
                auto &workspace = workspace_by_key[PhysicalKey{
                    participant.world_rank, participant.device}];
                workspace = checkedAdd(
                    workspace,
                    maintenance_workspace_bytes,
                    "device rebalance maintenance workspace");
            }
        }

        std::map<PhysicalKey, std::size_t> budget_by_key;
        std::map<std::string, std::size_t> budget_by_id;
        // CPU expert bytes price through the overlay's inventory-bound
        // uniform layout. A CPU resource without a resolved observation means
        // the plan was never hardware-bound through the refusal gate; admit
        // nothing rather than pricing an assumed layout.
        const bool has_cpu_resource = std::any_of(
            physical_budgets.begin(), physical_budgets.end(),
            [](const MoEOverlayBoundPhysicalMemoryBudget &budget)
            { return budget.device().is_cpu(); });
        std::optional<cpu::native_quantised::CPUNativeQuantisedLayout>
            resolved_observation = plan.resolved_cpu_prepared_layout;
        if (has_cpu_resource && !resolved_observation)
        {
            if (inventory == nullptr)
                throw std::invalid_argument(
                    "ExpertOverlay capacity admission requires the inventory-bound "
                    "uniform CPU prepared layout for every CPU resource");
            // Producers that reach admission with live inventory (pipeline
            // admission, candidate admission, the serving runner) derive the
            // observation here with the SAME uniformity refusal the hardware
            // binder applies; only API misuse without inventory keeps the
            // hard refusal above. The throw is never silently defaulted.
            resolved_observation =
                resolveUniformCpuPreparedLayoutFromInventory(*inventory, plan);
        }
        // The kHostPreparedLayout fallback below is dead for CPU byte pricing:
        // it is consumed only by host-only CPU budgets whose plan declares no
        // CPU participant (the derive returns nullopt exactly then), so no CPU
        // byte is ever priced with the ambient build layout. A CPU participant
        // always yields a real observation, asserted here as the invariant.
        if (has_cpu_resource && !resolved_observation)
        {
            LLAMINAR_ASSERT(
                !planDeclaresCpuParticipant(plan),
                "a CPU-participant plan must carry a resolved CPU layout "
                "observation before pricing");
        }
        MoEOverlayCapacityResolverInput input;
        input.num_experts = num_experts;
        input.cpu_prepared_layout = resolved_observation.value_or(
            cpu::native_quantised::kHostPreparedLayout);
        input.initial_residency_policy =
            policy.migrationEnabled()
                ? MoEOverlayInitialResidencyPolicy::
                      MigrationSourcePerParticipant
                : MoEOverlayInitialResidencyPolicy::PriorityFillOnly;
        input.layer_weight_manifest = layer_weight_manifest;
        input.physical_budgets.reserve(physical_budgets.size());
        for (const auto &budget : physical_budgets)
        {
            const PhysicalKey key{budget.worldRank(), budget.device()};
            if (budget.worldRank() < 0 || !budget.device().is_valid() ||
                budget.resourceId().empty() ||
                !budget_by_key.emplace(key, input.physical_budgets.size()).second ||
                !budget_by_id.emplace(
                     budget.resourceId(), input.physical_budgets.size()).second)
            {
                throw std::invalid_argument(
                    "ExpertOverlay bound physical budgets require unique rank/device and resource identities");
            }
            const auto staging = staging_by_key.find(key);
            const std::size_t canonical_staging =
                staging == staging_by_key.end() ? 0 : staging->second;
            PhysicalMemoryBOMBuilder complete_builder(
                budget.bom());
            complete_builder.add(
                PhysicalMemoryOwner::ExpertMigrationStaging,
                canonical_staging);
            const auto workspace = workspace_by_key.find(key);
            complete_builder.add(
                PhysicalMemoryOwner::ExecutionWorkspace,
                workspace == workspace_by_key.end()
                    ? 0u
                    : workspace->second);
            const auto complete_bom = complete_builder.build();
            if (!complete_bom.fits())
                throw MoEOverlayCapacityExhausted(
                    "ExpertOverlay fixed migration BOM exceeds " +
                    budget.resourceId() + ": deficit_bytes=" +
                    std::to_string(complete_bom.deficitBytes()));
            input.physical_budgets.emplace_back(
                budget.resourceId(),
                PhysicalMemoryAdmissionCertificate(complete_bom));
        }
        if (input.physical_budgets.empty())
        {
            throw std::invalid_argument(
                "ExpertOverlay capacity admission requires physical budgets");
        }
        for (const auto &[key, bytes] : staging_by_key)
        {
            (void)bytes;
            if (budget_by_key.count(key) == 0)
            {
                std::ostringstream error;
                error << "ExpertOverlay migration staging at rank "
                      << key.world_rank << " on " << key.device.toString()
                      << " has no physical memory authority";
                throw std::invalid_argument(error.str());
            }
        }
        for (const auto &[key, bytes] : workspace_by_key)
        {
            (void)bytes;
            if (budget_by_key.count(key) == 0)
            {
                std::ostringstream error;
                error << "ExpertOverlay maintenance workspace at rank "
                      << key.world_rank << " on " << key.device.toString()
                      << " has no physical memory authority";
                throw std::invalid_argument(error.str());
            }
        }

        std::size_t bound_participant_cursor = 0;
        input.tiers.reserve(plan.routed_tiers.size());
        for (std::size_t tier_index = 0;
             tier_index < plan.routed_tiers.size();
             ++tier_index)
        {
            const auto &tier = plan.routed_tiers[tier_index];
            const auto &domain = requireDomain(plan, tier);
            if (domain.participants.empty())
            {
                throw std::invalid_argument(
                    "ExpertOverlay capacity domain '" + domain.name +
                    "' has no participants");
            }

            MoEOverlayTierCapacityRequest request;
            request.tier_index = static_cast<int>(tier_index);
            request.tier_name = tier.name;
            request.priority = tier.priority;
            request.fallback = tier.fallback;
            request.copy_policy = copyPolicy(domain);
            if (request.copy_policy == MoEOverlayTierCopyPolicy::GateUpOwnedDownColumns &&
                (plan.routed_tiers.size() != 1u || policy.usesPhysicalResidencyFabric()))
                throw std::invalid_argument(
                    "Gate/up-owned down columns require a single native GPU domain; cross-tier slice publication is not implemented");
            request.quota_mode =
                tier.resolved_live_experts_per_layer.empty()
                    ? MoEOverlayLiveQuotaMode::Automatic
                    : MoEOverlayLiveQuotaMode::FixedPerLayer;
            request.fixed_live_experts_per_layer =
                tier.resolved_live_experts_per_layer;

            std::vector<DeviceId> participant_devices;
            participant_devices.reserve(domain.participants.size());
            request.participants.reserve(domain.participants.size());
            for (std::size_t participant_index = 0;
                 participant_index < domain.participants.size();
                 ++participant_index)
            {
                if (bound_participant_cursor >= bound_participants.size())
                    throw std::logic_error("Bound ExpertOverlay participant enumeration truncated");
                const auto &bound =
                    bound_participants[bound_participant_cursor++];
                if (bound.tier_index != static_cast<int>(tier_index) ||
                    bound.domain_participant_index !=
                        static_cast<int>(participant_index))
                {
                    throw std::logic_error(
                        "Bound ExpertOverlay participant order diverged from tier topology");
                }
                const DeviceId device = bound.device;
                const int world_rank = bound.world_rank;
                const auto budget = budget_by_key.find(
                    PhysicalKey{world_rank, device});
                if (budget == budget_by_key.end())
                {
                    std::ostringstream error;
                    error << "ExpertOverlay tier '" << tier.name
                          << "' participant " << participant_index
                          << " at rank " << world_rank << " on "
                          << device.toString()
                          << " has no physical capacity authority";
                    throw std::invalid_argument(error.str());
                }

                request.participants.push_back({
                    .participant_id = bound.participant_id,
                    .resource_id = input.physical_budgets[budget->second]
                                       .resourceId(),
                    .shadow_slots_per_layer =
                        policy.usesPhysicalResidencyFabric()
                            ? policy.shadow_slots_per_endpoint_layer
                            : 0u,
                    .maximum_concurrent_shadow_slots =
                        policy.usesPhysicalResidencyFabric()
                            ? policy.maximum_concurrent_cycles
                            : 0u,
                });
                participant_devices.push_back(device);
            }

            if(request.copy_policy==MoEOverlayTierCopyPolicy::TensorSharded){
                if(domain.scope!=ExecutionDomainScope::NODE_LOCAL ||
                   domain.world_ranks.size()!=domain.participants.size())
                    throw std::invalid_argument("CPU tensor-shard admission requires original node-local ranks");
                std::vector<MoEExpertShardCohort::Participant> members;
                for(std::size_t p=0;p<domain.participants.size();++p)
                    members.push_back({request.participants[p].participant_id,
                        boundParticipantWorldRank(domain,p),domain.participants[p]});
                for(const auto& layer:layer_weight_manifest)
                    request.shard_cohorts.push_back(std::make_shared<const MoEExpertShardCohort>(
                        MoEExpertShardCohort::checked(layer,num_experts,members,
                            MoEExpertShardCohort::balancedIntervals(layer,static_cast<int>(members.size())))));
                if(tier.memory_budget_bytes!=0)
                    throw std::invalid_argument("CPU shard tiers require physical PMA budgets, not an ambiguous aggregate tier byte cap");
            }
            request.max_live_experts_per_layer.assign(
                layer_count, num_experts);
            const auto tier_footprints =
                MoEOverlayCapacityResolver::preparedFootprints(
                    layer_weight_manifest, moeOverlayMovableProjections(request.copy_policy));
            for (std::size_t layer = 0; layer < layer_count; ++layer)
            {
                int maximum = num_experts;
                if (tier.max_experts_per_layer > 0)
                {
                    maximum = std::min(
                        maximum, tier.max_experts_per_layer);
                }
                maximum = std::min(
                    maximum,
                    exactLegacyLayerCap(
                        tier.memory_budget_bytes,
                        num_experts,
                        layer,
                        request.copy_policy,
                        participant_devices,
                        tier_footprints));
                request.max_live_experts_per_layer[layer] = maximum;
            }
            input.tiers.push_back(std::move(request));
        }

        return input;
    }

    MoEOverlayResolvedCapacityPlan
    MoEOverlayCapacityAdmission::resolveCapacity(
        const MoERoutedExpertPlacementPlan &plan,
        int num_experts,
        const std::vector<MoEOverlayLayerWeightManifest> &layer_weight_manifest,
        const std::vector<MoEOverlayBoundPhysicalMemoryBudget> &physical_budgets,
        const MoEOverlayCapacityAdmissionPolicy &policy,
        const ClusterInventory *inventory)
    {
        const MoEOverlayStageCapacityRequest stage{
            plan, num_experts, layer_weight_manifest, physical_budgets, policy};
        auto result = resolvePipelineCapacity(
            std::span(&stage, 1u), inventory);
        return std::move(result.front());
    }

    std::vector<MoEOverlayResolvedCapacityPlan>
    MoEOverlayCapacityAdmission::resolvePipelineCapacity(
        std::span<const MoEOverlayStageCapacityRequest> stages,
        const ClusterInventory *inventory)
    {
        return resolvePipelineCapacityDraft(stages, inventory).admit();
    }

    MoEOverlayCapacityDraft MoEOverlayCapacityAdmission::resolveCapacityDraft(
        const MoERoutedExpertPlacementPlan &plan,int num_experts,
        const std::vector<MoEOverlayLayerWeightManifest> &manifest,
        const std::vector<MoEOverlayBoundPhysicalMemoryBudget> &budgets,
        const MoEOverlayCapacityAdmissionPolicy &policy,
        const ClusterInventory *inventory)
    {
        const MoEOverlayStageCapacityRequest stage{plan,num_experts,manifest,budgets,policy};
        return resolvePipelineCapacityDraft(
            std::span(&stage,1u), inventory);
    }

    MoEOverlayCapacityDraft MoEOverlayCapacityAdmission::resolvePipelineCapacityDraft(
        std::span<const MoEOverlayStageCapacityRequest> stages,
        const ClusterInventory *inventory)
    {
        if (stages.empty())
            throw std::invalid_argument("ExpertOverlay pipeline admission requires at least one stage");

        std::vector<std::uint32_t> requested(stages.size(), 0u);
        std::vector<std::uint32_t> minimum(stages.size(), 0u);
        for (std::size_t index = 0; index < stages.size(); ++index)
        {
            const auto &stage = stages[index];
            if (!stage.policy.usesDeviceTransferDirectory()) continue;
            if (!stage.policy.device_rebalance_workspace_capacity)
                throw std::invalid_argument("Native replica admission requires its workspace capacity");
            const auto &workspace = *stage.policy.device_rebalance_workspace_capacity;
            const auto maximum = workspace.max_hot_replicas_per_participant;
            if (maximum > static_cast<std::uint32_t>(std::max(0, stage.num_experts)))
                throw std::invalid_argument("Replica-cache upper bound exceeds the model expert count");
            const auto &directory = stage.policy.device_transfer_directory_capacity;
            if (directory.active_slots != static_cast<std::uint64_t>(workspace.num_layers) * std::max(1u, maximum) ||
                static_cast<std::uint64_t>(directory.active_slots) + directory.staging_slots != directory.total_slots ||
                workspace.local_transfer_slot_count != directory.total_slots)
                throw std::invalid_argument("Replica admission requires exact requested directory geometry");
            requested[index] = maximum;
            if (stage.plan.replica_cache_capacity)
            {
                if (maximum != static_cast<std::uint32_t>(stage.plan.replica_cache_capacity->admitted()))
                    throw std::logic_error("Retained replica-cache policy differs from its admitted grant");
                minimum[index] = maximum;
            }
            else
                minimum[index] = std::min(1u, maximum);
        }

        // Every candidate is a complete pipeline BOM. Failed candidates own no
        // allocations, and a successful result has one certificate for all stages.
        const auto resolve_exact = [&](const std::vector<std::uint32_t> &grants)
        {
            std::vector<MoEOverlayCapacityResolverInput> inputs;
            inputs.reserve(stages.size());
            for (std::size_t index = 0; index < stages.size(); ++index)
            {
                const auto &stage = stages[index];
                auto candidate = stage.policy;
                if (candidate.usesDeviceTransferDirectory())
                {
                    auto &workspace = *candidate.device_rebalance_workspace_capacity;
                    workspace.max_hot_replicas_per_participant = grants[index];
                    // Persistent replicas scale independently of the admitted
                    // rolling transfer pool and its actual communication extent.
                    candidate.device_transfer_directory_capacity = DeviceMoETransferSlotDirectory::planBufferedCapacity(
                        static_cast<std::uint64_t>(workspace.num_layers) * std::max(1u, grants[index]),
                        stage.policy.device_transfer_directory_capacity.staging_slots,
                        /*one already-combined staging pool=*/1u);
                    workspace.local_transfer_slot_count = candidate.device_transfer_directory_capacity.total_slots;
                    workspace.collective_payload_slot_capacity = std::min(
                        stage.policy.device_rebalance_workspace_capacity->collective_payload_slot_capacity,
                        workspace.local_transfer_slot_count);
                }
                inputs.push_back(buildResolverInput(stage.plan, stage.num_experts,
                    stage.layer_weight_manifest, stage.physical_budgets,
                    candidate, inventory));
            }
            auto result = MoEOverlayCapacityResolver::resolvePipelineDraft(inputs);
            std::vector<std::optional<MoEOverlayReplicaCacheCapacity>> admitted(stages.size());
            for (std::size_t index=0;index<stages.size();++index)
                if(stages[index].policy.usesDeviceTransferDirectory())
                    admitted[index]=stages[index].plan.replica_cache_capacity
                        ? stages[index].plan.replica_cache_capacity
                        : std::optional<MoEOverlayReplicaCacheCapacity>{std::in_place,requested[index],grants[index]};
            return result.withReplicaGrants(admitted);
        };

        try
        {
            return resolve_exact(requested);
        }
        catch (const MoEOverlayCapacityExhausted &)
        {
            if (requested == minimum) throw;
        }
        // An enabled cache keeps its minimum on every stage. This is a capacity
        // search within the selected mode, never a change of execution policy.
        auto grants = minimum;
        auto best = resolve_exact(grants);
        for (std::size_t index = 0; index < stages.size(); ++index)
        {
            if (requested[index] == minimum[index]) continue;
            std::uint32_t low = minimum[index] + 1u;
            std::uint32_t high = requested[index];
            while (low <= high)
            {
                const auto middle = low + (high - low) / 2u;
                auto candidate = grants;
                candidate[index] = middle;
                try
                {
                    auto admitted = resolve_exact(candidate);
                    grants = std::move(candidate);
                    best = std::move(admitted);
                    low = middle + 1u;
                }
                catch (const MoEOverlayCapacityExhausted &)
                {
                    high = middle - 1u;
                }
            }
        }
        return best;
    }

    MoERoutedExpertPlacementPlan
    MoEOverlayCapacityAdmission::resolveAndInstall(
        const MoERoutedExpertPlacementPlan &plan,
        int num_experts,
        const std::vector<MoEOverlayLayerWeightManifest> &layer_weight_manifest,
        const std::vector<MoEOverlayBoundPhysicalMemoryBudget> &physical_budgets,
        const MoEOverlayCapacityAdmissionPolicy &policy,
        const ClusterInventory *inventory)
    {
        const auto capacity = resolveCapacity(
            plan,
            num_experts,
            layer_weight_manifest,
            physical_budgets,
            policy,
            inventory);
        return MoEOverlayCapacityResolver::installResolvedQuotas(
            plan, capacity);
    }
} // namespace llaminar2
