/**
 * @file MoEOverlayCapacityAdmission.cpp
 * @brief Hardware-bound ExpertOverlay capacity admission implementation.
 *
 * Logical participants are joined to physical memory through typed rank/device
 * keys before the pure byte resolver runs.  All policy decisions come from
 * scoped enums and integer priorities. Opaque tier and domain labels are used
 * solely for identity and diagnostics.
 */

#include "MoEOverlayCapacityAdmission.h"

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
            case RoutedExpertComputePolicy::Replicated:
                return MoEOverlayTierCopyPolicy::Replicated;
            case RoutedExpertComputePolicy::TensorSharded:
                throw std::invalid_argument(
                    "ExpertOverlay exact whole-expert capacity does not yet "
                    "admit tensor-sharded routed domain '" + domain.name + "'");
            case RoutedExpertComputePolicy::Unspecified:
                throw std::invalid_argument(
                    "ExpertOverlay capacity domain '" + domain.name +
                    "' has an unspecified routed compute policy");
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
                if (policy == MoEOverlayTierCopyPolicy::Apportioned)
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

    std::vector<MoEOverlayTransferStagingCharge>
    MoEOverlayCapacityAdmission::transferStagingBOM(
        const MoERoutedExpertPlacementPlan &plan,
        int world_size,
        const MoEOverlayCapacityAdmissionPolicy &policy)
    {
        if (!policy.materialize_migration_fabric)
            return {};
        if (world_size <= 0 || policy.staging_capacity_bytes == 0 ||
            policy.shadow_slots_per_endpoint_layer == 0 ||
            policy.maximum_concurrent_cycles == 0 ||
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

        constexpr std::size_t kProjectionCount = 3;
        constexpr std::size_t kRemoteRolesPerGpu = 2;
        constexpr std::size_t kBlobSlots = 2;
        constexpr std::size_t kPinnedRegionsPerBlobSlot = 2;

        const auto participants = boundParticipants(plan);
        std::map<int, std::map<DeviceId, std::size_t>>
            participant_multiplicity_by_rank;
        for (const auto &participant : participants)
        {
            if (participant.world_rank < 0 ||
                participant.world_rank >= world_size)
            {
                throw std::invalid_argument(
                    "ExpertOverlay participant rank lies outside the overlay communicator");
            }
            ++participant_multiplicity_by_rank[participant.world_rank]
                                               [participant.device];
        }

        std::map<PhysicalKey, std::size_t> bytes_by_resource;
        const auto addChunks = [&](
            int world_rank,
            DeviceId device,
            std::size_t chunks,
            const char *what)
        {
            const std::size_t bytes = checkedMultiply(
                chunks, policy.staging_capacity_bytes, what);
            auto &total = bytes_by_resource[PhysicalKey{world_rank, device}];
            total = checkedAdd(total, bytes, what);
        };

        for (int world_rank = 0; world_rank < world_size; ++world_rank)
        {
            const auto found = participant_multiplicity_by_rank.find(
                world_rank);
            const std::map<DeviceId, std::size_t> empty;
            const auto &devices = found ==
                    participant_multiplicity_by_rank.end()
                ? empty
                : found->second;

            /*
             * A closed cycle visits a logical participant at most once. Thus
             * one physical edge can carry at most the smaller endpoint
             * multiplicity per cycle. Materialize that exact upper bound for
             * every admitted cycle so software never queues one independent
             * move behind another.
             */
            for (const auto &[source, source_multiplicity] : devices)
            {
                for (const auto &[destination, destination_multiplicity] :
                     devices)
                {
                    if (source == destination)
                        continue;
                    const std::size_t edge_lanes = checkedMultiply(
                        policy.maximum_concurrent_cycles,
                        std::min(
                            source_multiplicity,
                            destination_multiplicity),
                        "local parallel migration lanes");
                    if ((source.is_cpu() && destination.is_gpu()) ||
                        (source.is_gpu() && destination.is_cpu()))
                    {
                        const DeviceId gpu =
                            source.is_gpu() ? source : destination;
                        addChunks(
                            world_rank,
                            gpu,
                            checkedMultiply(
                                kProjectionCount,
                                edge_lanes,
                                "local GPU/CPU projection lanes"),
                            "local GPU/CPU device staging");
                        addChunks(
                            world_rank,
                            DeviceId::cpu(),
                            checkedMultiply(
                                kProjectionCount,
                                edge_lanes,
                                "local GPU/CPU projection lanes"),
                            "local GPU/CPU pinned staging");
                    }
                    else if (source.is_gpu() && destination.is_gpu() &&
                             source.type != destination.type)
                    {
                        /* Two slots, each with source- and destination-runtime pinned storage. */
                        addChunks(
                            world_rank,
                            DeviceId::cpu(),
                            checkedMultiply(
                                kProjectionCount,
                                checkedMultiply(
                                    edge_lanes,
                                    kBlobSlots *
                                        kPinnedRegionsPerBlobSlot,
                                    "heterogeneous GPU parallel lanes"),
                                "heterogeneous GPU blob staging chunks"),
                            "heterogeneous GPU blob pinned staging");
                    }
                }
            }

            if (!policy.distributed_transport)
                continue;

            for (const auto &[device, multiplicity] : devices)
            {
                if (!device.is_gpu())
                    continue;
                const std::size_t remote_chunks = checkedMultiply(
                    checkedMultiply(
                        kProjectionCount,
                        kRemoteRolesPerGpu,
                        "remote GPU projection roles"),
                    checkedMultiply(
                        policy.maximum_concurrent_cycles,
                        multiplicity,
                        "remote GPU parallel lanes"),
                    "remote GPU staging chunks");
                addChunks(
                    world_rank,
                    device,
                    remote_chunks,
                    "remote GPU device staging");
                addChunks(
                    world_rank,
                    DeviceId::cpu(),
                    remote_chunks,
                    "remote GPU pinned staging");
            }

            /* The MPI data plane retains one host payload per global projection lane. */
            addChunks(
                world_rank,
                DeviceId::cpu(),
                checkedMultiply(
                    participants.size(),
                    checkedMultiply(
                        kProjectionCount,
                        policy.maximum_concurrent_cycles,
                        "remote MPI concurrent projection lanes"),
                    "remote MPI projection lanes"),
                "remote MPI payload staging");
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
        const MoEOverlayCapacityAdmissionPolicy &policy)
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
        if (policy.materialize_migration_fabric &&
            (policy.shadow_slots_per_endpoint_layer == 0 ||
             policy.staging_capacity_bytes == 0 ||
             policy.maximum_concurrent_cycles == 0 ||
             policy.maximum_cycles_per_layer == 0 ||
             policy.shadow_slots_per_endpoint_layer <
                 MoEOverlayCapacityAdmissionPolicy::
                     requiredShadowSlotsPerEndpointLayer(
                         policy.maximum_concurrent_cycles,
                         policy.maximum_cycles_per_layer)))
        {
            throw std::invalid_argument(
                "ExpertOverlay migration admission requires staging and one endpoint/layer shadow slot per admitted same-layer cycle");
        }

        const auto footprints =
            MoEOverlayCapacityResolver::preparedFootprints(
                layer_weight_manifest);
        const std::size_t layer_count = footprints.size();

        const auto bound_participants = boundParticipants(plan);
        const auto staging_bom = transferStagingBOM(
            plan, policy.overlay_world_size, policy);
        std::map<PhysicalKey, std::size_t> staging_by_key;
        for (const auto &charge : staging_bom)
        {
            staging_by_key.emplace(
                PhysicalKey{charge.world_rank, charge.device},
                charge.bytes);
        }

        std::map<PhysicalKey, std::size_t> budget_by_key;
        std::map<std::string, std::size_t> budget_by_id;
        MoEOverlayCapacityResolverInput input;
        input.num_experts = num_experts;
        input.initial_residency_policy =
            policy.materialize_migration_fabric
                ? MoEOverlayInitialResidencyPolicy::
                      MigrationSourcePerParticipant
                : MoEOverlayInitialResidencyPolicy::PriorityFillOnly;
        input.layer_weight_manifest = layer_weight_manifest;
        input.physical_budgets.reserve(physical_budgets.size());
        for (const auto &budget : physical_budgets)
        {
            const PhysicalKey key{budget.world_rank, budget.device};
            if (budget.world_rank < 0 || !budget.device.is_valid() ||
                budget.resource_id.empty() ||
                !budget_by_key.emplace(key, input.physical_budgets.size()).second ||
                !budget_by_id.emplace(
                     budget.resource_id, input.physical_budgets.size()).second)
            {
                throw std::invalid_argument(
                    "ExpertOverlay bound physical budgets require unique rank/device and resource identities");
            }
            const auto staging = staging_by_key.find(key);
            const std::size_t canonical_staging =
                staging == staging_by_key.end() ? 0 : staging->second;
            input.physical_budgets.push_back({
                .resource_id = budget.resource_id,
                .device = budget.device,
                .usable_budget_bytes = budget.usable_budget_bytes,
                .fixed_bytes = budget.fixed_bytes,
                .transfer_staging_bytes = checkedAdd(
                    canonical_staging,
                    budget.additional_transfer_staging_bytes,
                    "physical transfer staging"),
            });
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
                                       .resource_id,
                    .shadow_slots_per_layer =
                        policy.shadow_slots_per_endpoint_layer,
                });
                participant_devices.push_back(device);
            }

            request.max_live_experts_per_layer.assign(
                layer_count, num_experts);
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
                        footprints));
                request.max_live_experts_per_layer[layer] = maximum;
            }
            input.tiers.push_back(std::move(request));
        }

        return input;
    }

    MoERoutedExpertPlacementPlan
    MoEOverlayCapacityAdmission::resolveAndInstall(
        const MoERoutedExpertPlacementPlan &plan,
        int num_experts,
        const std::vector<MoEOverlayLayerWeightManifest> &layer_weight_manifest,
        const std::vector<MoEOverlayBoundPhysicalMemoryBudget> &physical_budgets,
        const MoEOverlayCapacityAdmissionPolicy &policy)
    {
        const auto input = buildResolverInput(
            plan,
            num_experts,
            layer_weight_manifest,
            physical_budgets,
            policy);
        const auto capacity = MoEOverlayCapacityResolver::resolve(input);
        return MoEOverlayCapacityResolver::installResolvedQuotas(
            plan, capacity);
    }
} // namespace llaminar2
