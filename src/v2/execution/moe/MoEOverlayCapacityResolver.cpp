/**
 * @file MoEOverlayCapacityResolver.cpp
 * @brief Exact prepared-weight BOM and integer-priority quota resolution.
 *
 * All arithmetic in this file is setup-time and fail-closed.  The implementation
 * mirrors the allocation layouts used by CpuExpertSlotPool and
 * GpuExpertSlotPool, then admits logical residents one at a time while charging
 * their actual physical copy multiplicity.  No runtime allocator or backend is
 * consulted, which keeps the same decision reproducible on every MPI rank.
 */

#include "MoEOverlayCapacityResolver.h"

#include "ExpertPreparedMemoryGeometry.h"

#include <algorithm>
#include <limits>
#include <map>
#include <numeric>
#include <set>
#include <sstream>
#include <stdexcept>
#include <unordered_map>
#include <utility>

namespace llaminar2
{
    namespace
    {
        [[nodiscard]] std::size_t checkedAdd(
            std::size_t lhs,
            std::size_t rhs,
            const char *what)
        {
            if (rhs > std::numeric_limits<std::size_t>::max() - lhs)
                throw std::overflow_error(std::string(what) + " byte sum overflows size_t");
            return lhs + rhs;
        }

        [[nodiscard]] std::size_t checkedMultiply(
            std::size_t lhs,
            std::size_t rhs,
            const char *what)
        {
            if (lhs != 0 && rhs > std::numeric_limits<std::size_t>::max() / lhs)
                throw std::overflow_error(std::string(what) + " byte product overflows size_t");
            return lhs * rhs;
        }

        /** @brief Mutable setup-only extension of one certified base BOM. */
        struct ResourceState
        {
            std::string resource_id;
            PhysicalMemoryBOMBuilder memory;
            std::vector<int> live_copies_per_layer;
            std::vector<int> shadow_arrival_capacity_per_layer;

            ResourceState(
                std::string identity,
                const PhysicalMemoryAdmissionCertificate &base,
                std::size_t layer_count)
                : resource_id(std::move(identity)),
                  memory(base.bom()),
                  live_copies_per_layer(layer_count, 0),
                  shadow_arrival_capacity_per_layer(layer_count, 0)
            {
            }
        };

        struct TierState
        {
            const MoEOverlayTierCapacityRequest *request = nullptr;
            MoEOverlayResolvedTierCapacity output;
        };

        [[nodiscard]] std::size_t resourceUsed(
            const ResourceState &resource)
        {
            return resource.memory.build().incrementalBytes();
        }

        [[nodiscard]] bool resourceCanAdd(
            const ResourceState &resource,
            std::size_t bytes)
        {
            const auto bom = resource.memory.build();
            const std::size_t used = bom.incrementalBytes();
            const std::size_t available =
                bom.resource().admission_available_bytes;
            return used <= available && bytes <= available - used;
        }

        [[nodiscard]] std::size_t footprintFor(
            const MoEOverlayPreparedExpertFootprint &footprint,
            DeviceId device,
            bool shadow)
        {
            return shadow ? footprint.shadowBytes(device)
                          : footprint.liveBytes(device);
        }

        /** @brief Build the exact shared-arena key from one layer manifest. */
        [[nodiscard]] ExpertPreparedTripletGeometryKey geometryKey(
            const MoEOverlayLayerWeightManifest &layer)
        {
            ExpertPreparedTripletGeometryKey key;
            for (const auto &projection : layer.projections)
            {
                const auto role = static_cast<std::size_t>(
                    projection.projection);
                if (role >= key.projections.size())
                {
                    throw std::invalid_argument(
                        "ExpertOverlay layer manifest has an unknown projection role");
                }
                key.projections[role] = {
                    .N = projection.N,
                    .K = projection.K,
                    .format = projection.format,
                };
            }
            return key;
        }

        [[nodiscard]] std::map<std::string, std::size_t> tierShadowCharges(
            const TierState &tier,
            const std::vector<MoEOverlayPreparedExpertFootprint> &footprints,
            const std::vector<MoEOverlayLayerWeightManifest> &manifest,
            const std::unordered_map<std::string, std::size_t> &resource_index,
            const std::vector<ResourceState> &resources)
        {
            if (manifest.size() != footprints.size())
            {
                throw std::logic_error(
                    "ExpertOverlay shadow geometry and footprint inventories disagree");
            }
            std::map<ExpertPreparedTripletGeometryKey, std::vector<std::size_t>>
                layers_by_geometry;
            for (std::size_t layer = 0; layer < manifest.size(); ++layer)
                layers_by_geometry[geometryKey(manifest[layer])].push_back(layer);

            std::map<std::string, std::size_t> charges;
            for (const auto &participant : tier.request->participants)
            {
                const auto found = resource_index.find(participant.resource_id);
                if (found == resource_index.end())
                    throw std::logic_error("Validated ExpertOverlay resource disappeared");
                const DeviceId device =
                    resources[found->second].memory.build().resource().device;
                for (const auto &[_, layers] : layers_by_geometry)
                {
                    if (layers.empty() ||
                        layers.size() >
                            std::numeric_limits<std::size_t>::max() /
                                std::max<std::size_t>(
                                    1, participant.shadow_slots_per_layer))
                    {
                        throw std::overflow_error(
                            "ExpertOverlay shared shadow arena capacity overflows size_t");
                    }
                    const std::size_t slots = std::min(
                        participant.maximum_concurrent_shadow_slots,
                        layers.size() * participant.shadow_slots_per_layer);
                    const std::size_t bytes = checkedMultiply(
                        slots,
                        footprintFor(
                            footprints[layers.front()],
                            device,
                            /*shadow=*/true),
                        "tier shared-geometry shadow slots");
                    charges[participant.resource_id] = checkedAdd(
                        charges[participant.resource_id], bytes, "tier shadows");
                }
            }
            return charges;
        }

        [[nodiscard]] std::map<std::string, std::size_t> nextLiveCharges(
            const TierState &tier,
            int layer_idx,
            const std::vector<MoEOverlayPreparedExpertFootprint> &footprints,
            const std::unordered_map<std::string, std::size_t> &resource_index,
            const std::vector<ResourceState> &resources)
        {
            std::map<std::string, std::size_t> charges;
            const int quota =
                tier.output.live_experts_per_layer[static_cast<std::size_t>(layer_idx)];
            if (tier.request->copy_policy == MoEOverlayTierCopyPolicy::Apportioned)
            {
                const std::size_t participant_index =
                    static_cast<std::size_t>(quota) %
                    tier.request->participants.size();
                const auto &participant =
                    tier.request->participants[participant_index];
                const auto resource = resource_index.at(participant.resource_id);
                charges[participant.resource_id] = footprintFor(
                    footprints[static_cast<std::size_t>(layer_idx)],
                    resources[resource].memory.build().resource().device,
                    /*shadow=*/false);
                return charges;
            }

            for (const auto &participant : tier.request->participants)
            {
                const auto resource = resource_index.at(participant.resource_id);
                const std::size_t bytes = footprintFor(
                    footprints[static_cast<std::size_t>(layer_idx)],
                    resources[resource].memory.build().resource().device,
                    /*shadow=*/false);
                charges[participant.resource_id] = checkedAdd(
                    charges[participant.resource_id], bytes, "replicated live expert");
            }
            return charges;
        }

        [[nodiscard]] bool chargesFit(
            const std::map<std::string, std::size_t> &charges,
            const std::unordered_map<std::string, std::size_t> &resource_index,
            const std::vector<ResourceState> &resources)
        {
            return std::all_of(
                charges.begin(), charges.end(), [&](const auto &entry)
                {
                    return resourceCanAdd(
                        resources[resource_index.at(entry.first)], entry.second);
                });
        }

        void commitCharges(
            const std::map<std::string, std::size_t> &charges,
            const std::unordered_map<std::string, std::size_t> &resource_index,
            std::vector<ResourceState> &resources,
            bool shadow)
        {
            for (const auto &[resource_id, bytes] : charges)
            {
                auto &resource = resources[resource_index.at(resource_id)];
                resource.memory.add(
                    shadow ? PhysicalMemoryOwner::ExpertShadowSlots
                           : PhysicalMemoryOwner::RoutedExpertWeights,
                    bytes);
            }
        }

        [[nodiscard]] bool addOneLiveExpert(
            TierState &tier,
            int layer_idx,
            const std::vector<MoEOverlayPreparedExpertFootprint> &footprints,
            const std::unordered_map<std::string, std::size_t> &resource_index,
            std::vector<ResourceState> &resources)
        {
            const std::size_t layer = static_cast<std::size_t>(layer_idx);
            if (!tier.request->max_live_experts_per_layer.empty() &&
                tier.output.live_experts_per_layer[layer] >=
                    tier.request->max_live_experts_per_layer[layer])
            {
                return false;
            }
            const auto live_charges = nextLiveCharges(
                tier, layer_idx, footprints, resource_index, resources);
            if (!chargesFit(live_charges, resource_index, resources))
                return false;
            commitCharges(
                live_charges, resource_index, resources, /*shadow=*/false);

            const int previous_quota = tier.output.live_experts_per_layer[layer];
            ++tier.output.live_experts_per_layer[layer];
            tier.output.has_live_residency = true;
            if (tier.request->copy_policy == MoEOverlayTierCopyPolicy::Apportioned)
            {
                const std::size_t participant =
                    static_cast<std::size_t>(previous_quota) %
                    tier.request->participants.size();
                ++tier.output.participant_live_copies[participant][layer];
                auto &resource = resources[resource_index.at(
                    tier.request->participants[participant].resource_id)];
                ++resource.live_copies_per_layer[layer];
            }
            else
            {
                for (std::size_t participant = 0;
                     participant < tier.request->participants.size();
                     ++participant)
                {
                    ++tier.output.participant_live_copies[participant][layer];
                    auto &resource = resources[resource_index.at(
                        tier.request->participants[participant].resource_id)];
                    ++resource.live_copies_per_layer[layer];
                }
            }
            return true;
        }

        [[noreturn]] void throwTierCapacityFailure(
            const TierState &tier,
            int layer_idx,
            int requested_quota,
            const char *demand,
            const std::vector<MoEOverlayPreparedExpertFootprint> &footprints,
            const std::unordered_map<std::string, std::size_t> &resource_index,
            const std::vector<ResourceState> &resources)
        {
            std::ostringstream error;
            error << "ExpertOverlay tier '" << tier.request->tier_name
                  << "' cannot admit its " << demand
                  << " for layer " << layer_idx << ": requested "
                  << requested_quota << " experts under the complete physical BOM";

            const std::size_t layer = static_cast<std::size_t>(layer_idx);
            if (!tier.request->max_live_experts_per_layer.empty() &&
                tier.output.live_experts_per_layer[layer] >=
                    tier.request->max_live_experts_per_layer[layer])
            {
                error << "; resolved tier maximum="
                      << tier.request->max_live_experts_per_layer[layer]
                      << " experts";
            }

            /*
             * Report the exact next-copy charge against every affected
             * physical authority. Capacity failures happen before a result
             * plan exists, so this is the only place an operator can see
             * whether fixed graph state, live experts, or another concrete
             * category consumed the limiting bytes.
             */
            const auto charges = nextLiveCharges(
                tier,
                layer_idx,
                footprints,
                resource_index,
                resources);
            for (const auto &[resource_id, additional_bytes] : charges)
            {
                const auto &resource =
                    resources[resource_index.at(resource_id)];
                const auto bom = resource.memory.build();
                const std::size_t used = resourceUsed(resource);
                const std::size_t remaining =
                    used < bom.resource().admission_available_bytes
                        ? bom.resource().admission_available_bytes - used
                        : 0;
                const std::size_t staging =
                    bom.bytes(PhysicalMemoryOwner::WeightLoadStaging) +
                    bom.bytes(
                        PhysicalMemoryOwner::ActivationTransportStaging) +
                    bom.bytes(PhysicalMemoryOwner::ExpertMigrationStaging);
                error << "; resource='" << resource_id
                      << "' device=" << bom.resource().device.toString()
                      << " requires_additional_bytes=" << additional_bytes
                      << " remaining_bytes=" << remaining
                      << " usable_bytes="
                      << bom.resource().admission_available_bytes
                      << " fixed_bytes="
                      << (used - staging -
                          bom.bytes(PhysicalMemoryOwner::ExpertShadowSlots) -
                          bom.bytes(PhysicalMemoryOwner::RoutedExpertWeights))
                      << " staging_bytes="
                      << staging
                      << " shadow_bytes="
                      << bom.bytes(PhysicalMemoryOwner::ExpertShadowSlots)
                      << " live_expert_bytes="
                      << bom.bytes(PhysicalMemoryOwner::RoutedExpertWeights);
            }
            throw std::invalid_argument(error.str());
        }
    } // namespace

    std::size_t MoEOverlayPreparedExpertFootprint::liveBytes(
        DeviceId device) const
    {
        if (device.is_cpu())
            return cpu_live_bytes;
        if (device.is_gpu())
            return gpu_live_bytes;
        throw std::invalid_argument(
            "ExpertOverlay capacity footprint requires a valid CPU, CUDA, or ROCm device");
    }

    std::size_t MoEOverlayPreparedExpertFootprint::shadowBytes(
        DeviceId device) const
    {
        if (device.is_cpu())
            return cpu_shadow_bytes;
        if (device.is_gpu())
            return gpu_shadow_bytes;
        throw std::invalid_argument(
            "ExpertOverlay capacity footprint requires a valid CPU, CUDA, or ROCm device");
    }

    const MoEOverlayResolvedTierCapacity *MoEOverlayResolvedCapacityPlan::tier(
        int tier_index) const noexcept
    {
        const auto found = std::find_if(
            tiers.begin(), tiers.end(), [tier_index](const auto &entry)
            { return entry.tier_index == tier_index; });
        return found == tiers.end() ? nullptr : &*found;
    }

    const MoEOverlayResolvedPhysicalMemory *
    MoEOverlayResolvedCapacityPlan::resource(
        const std::string &resource_id) const noexcept
    {
        const auto found = std::find_if(
            physical_resources.begin(), physical_resources.end(),
            [&resource_id](const auto &entry)
            { return entry.resourceId() == resource_id; });
        return found == physical_resources.end() ? nullptr : &*found;
    }

    std::vector<MoEOverlayPreparedExpertFootprint>
    MoEOverlayCapacityResolver::preparedFootprints(
        const std::vector<MoEOverlayLayerWeightManifest> &manifest)
    {
        if (manifest.empty())
            throw std::invalid_argument("ExpertOverlay capacity resolver requires a layer manifest");

        std::vector<MoEOverlayPreparedExpertFootprint> footprints;
        footprints.reserve(manifest.size());
        for (std::size_t layer_offset = 0;
             layer_offset < manifest.size();
             ++layer_offset)
        {
            const auto &layer = manifest[layer_offset];
            if (!layer.valid() ||
                layer.layer_idx != static_cast<int>(layer_offset))
            {
                throw std::invalid_argument(
                    "ExpertOverlay capacity layer manifest must be valid, unique, and contiguous from zero");
            }

            MoEOverlayPreparedExpertFootprint footprint;
            footprint.layer_idx = layer.layer_idx;
            for (const auto &projection : layer.projections)
            {
                const auto geometry =
                    resolveExpertPreparedProjectionMemoryGeometry(
                        projection.N,
                        projection.K,
                        projection.format);
                footprint.cpu_live_bytes = checkedAdd(
                    footprint.cpu_live_bytes,
                    geometry.cpu_bytes,
                    "CPU live expert");
                footprint.cpu_shadow_bytes = checkedAdd(
                    footprint.cpu_shadow_bytes,
                    geometry.cpu_bytes,
                    "CPU shadow expert");
                footprint.gpu_live_bytes = checkedAdd(
                    footprint.gpu_live_bytes,
                    geometry.gpu_live_bytes,
                    "GPU live expert");
                footprint.gpu_shadow_bytes = checkedAdd(
                    footprint.gpu_shadow_bytes,
                    geometry.gpu_shadow_bytes,
                    "GPU shadow expert");
            }
            footprints.push_back(footprint);
        }
        return footprints;
    }

    MoEOverlayResolvedCapacityPlan MoEOverlayCapacityResolver::resolve(
        const MoEOverlayCapacityResolverInput &input)
    {
        if (input.num_experts <= 0)
            throw std::invalid_argument("ExpertOverlay capacity requires a positive expert count");
        if (input.initial_residency_policy !=
                MoEOverlayInitialResidencyPolicy::PriorityFillOnly &&
            input.initial_residency_policy !=
                MoEOverlayInitialResidencyPolicy::
                    MigrationSourcePerParticipant)
        {
            throw std::invalid_argument(
                "ExpertOverlay capacity received an unknown initial-residency policy");
        }
        const auto footprints = preparedFootprints(
            input.layer_weight_manifest);
        const std::size_t layer_count = footprints.size();
        if (input.physical_budgets.empty() || input.tiers.empty())
        {
            throw std::invalid_argument(
                "ExpertOverlay capacity requires physical budgets and routed tiers");
        }

        std::vector<ResourceState> resources;
        resources.reserve(input.physical_budgets.size());
        std::unordered_map<std::string, std::size_t> resource_index;
        for (const auto &budget : input.physical_budgets)
        {
            const auto &base_bom = budget.certificate().bom();
            if (budget.resourceId().empty() ||
                !budget.device().is_valid() ||
                (!budget.device().is_cpu() && !budget.device().is_gpu()) ||
                base_bom.resource().admission_available_bytes == 0 ||
                !resource_index.emplace(
                     budget.resourceId(), resources.size()).second)
            {
                throw std::invalid_argument(
                    "ExpertOverlay physical budgets require unique non-empty identities, valid devices, and positive usable bytes");
            }
            resources.emplace_back(
                budget.resourceId(), budget.certificate(), layer_count);
        }

        std::set<int> tier_indices;
        std::set<int> priorities;
        std::set<int> participant_ids;
        int fallback_count = 0;
        int fallback_priority = std::numeric_limits<int>::min();
        std::vector<TierState> tiers;
        tiers.reserve(input.tiers.size());
        for (const auto &request : input.tiers)
        {
            if (request.tier_index < 0 || request.tier_name.empty() ||
                request.participants.empty() ||
                !tier_indices.insert(request.tier_index).second ||
                !priorities.insert(request.priority).second)
            {
                throw std::invalid_argument(
                    "ExpertOverlay tiers require unique indices, names, strict priorities, and participants");
            }
            if (request.fallback)
            {
                ++fallback_count;
                fallback_priority = request.priority;
            }
            if (request.quota_mode == MoEOverlayLiveQuotaMode::Automatic)
            {
                if (!request.fixed_live_experts_per_layer.empty())
                {
                    throw std::invalid_argument(
                        "Automatic ExpertOverlay tier cannot carry fixed per-layer quotas");
                }
            }
            else if (request.fixed_live_experts_per_layer.size() != layer_count)
            {
                throw std::invalid_argument(
                    "Fixed ExpertOverlay tier requires exactly one quota per model layer");
            }
            if (!request.max_live_experts_per_layer.empty() &&
                request.max_live_experts_per_layer.size() != layer_count)
            {
                throw std::invalid_argument(
                    "ExpertOverlay tier maximum requires exactly one cap per model layer");
            }
            for (std::size_t layer = 0;
                 layer < request.max_live_experts_per_layer.size();
                 ++layer)
            {
                const int maximum = request.max_live_experts_per_layer[layer];
                if (maximum < 0 || maximum > input.num_experts ||
                    (request.quota_mode ==
                         MoEOverlayLiveQuotaMode::FixedPerLayer &&
                     request.fixed_live_experts_per_layer[layer] > maximum))
                {
                    throw std::invalid_argument(
                        "ExpertOverlay tier per-layer maximum is invalid or smaller than its fixed quota");
                }
            }

            for (const auto &participant : request.participants)
            {
                if (participant.participant_id < 0 ||
                    participant.resource_id.empty() ||
                    resource_index.count(participant.resource_id) == 0 ||
                    !participant_ids.insert(participant.participant_id).second ||
                    ((participant.shadow_slots_per_layer == 0) !=
                     (participant.maximum_concurrent_shadow_slots == 0)) ||
                    participant.maximum_concurrent_shadow_slots <
                        participant.shadow_slots_per_layer)
                {
                    throw std::invalid_argument(
                        "ExpertOverlay capacity participants require unique ids, known resources, and a coherent shared shadow capacity");
                }
            }

            TierState state;
            state.request = &request;
            state.output.tier_index = request.tier_index;
            state.output.tier_name = request.tier_name;
            state.output.priority = request.priority;
            state.output.fallback = request.fallback;
            state.output.live_experts_per_layer.assign(layer_count, 0);
            state.output.participant_live_copies.assign(
                request.participants.size(),
                std::vector<int>(layer_count, 0));
            tiers.push_back(std::move(state));
        }
        if (fallback_count != 1 ||
            std::any_of(tiers.begin(), tiers.end(), [&](const auto &tier)
            {
                return !tier.request->fallback &&
                       tier.request->priority > fallback_priority;
            }))
        {
            throw std::invalid_argument(
                "ExpertOverlay requires exactly one fallback tier with the greatest strict priority");
        }

        std::sort(tiers.begin(), tiers.end(), [](const auto &lhs, const auto &rhs)
        {
            return lhs.request->priority < rhs.request->priority;
        });

        /*
         * The physical fabric materializes every configured endpoint/layer
         * shadow bank during model setup, even when that tier initially owns no
         * live expert.  Charge the complete topology before admitting any live
         * quota so an empty tier cannot hide an overcommit. Aggregating
         * first also makes shared-resource accounting independent of tier order.
         */
        std::map<std::string, std::size_t> all_shadow_charges;
        for (const auto &tier : tiers)
        {
            const auto tier_charges = tierShadowCharges(
                tier,
                footprints,
                input.layer_weight_manifest,
                resource_index,
                resources);
            for (const auto &[resource_id, bytes] : tier_charges)
            {
                all_shadow_charges[resource_id] = checkedAdd(
                    all_shadow_charges[resource_id],
                    bytes,
                    "all tier shadows");
            }
        }
        if (!chargesFit(all_shadow_charges, resource_index, resources))
        {
            throw std::invalid_argument(
                "ExpertOverlay endpoint shadow banks exceed the complete physical BOM");
        }
        commitCharges(
            all_shadow_charges, resource_index, resources, /*shadow=*/true);
        for (const auto &tier : tiers)
        {
            for (const auto &participant : tier.request->participants)
            {
                auto &resource =
                    resources[resource_index.at(participant.resource_id)];
                for (std::size_t layer = 0; layer < layer_count; ++layer)
                {
                    if (participant.shadow_slots_per_layer >
                        static_cast<std::size_t>(
                            std::numeric_limits<int>::max() -
                            resource.shadow_arrival_capacity_per_layer[layer]))
                    {
                        throw std::overflow_error(
                            "ExpertOverlay shadow copy count exceeds int");
                    }
                    resource.shadow_arrival_capacity_per_layer[layer] +=
                        static_cast<int>(participant.shadow_slots_per_layer);
                }
            }
        }

        std::vector<int> unassigned(
            layer_count, input.num_experts);

        /*
         * Fixed quotas are exact constraints, so reserve and charge them before
         * any automatic tier consumes remaining capacity. This prevents a more-preferred
         * automatic tier from shrinking an explicitly requested later tier.
         */
        for (auto &tier : tiers)
        {
            if (tier.request->quota_mode !=
                MoEOverlayLiveQuotaMode::FixedPerLayer)
            {
                continue;
            }
            for (std::size_t layer = 0; layer < layer_count; ++layer)
            {
                const int quota =
                    tier.request->fixed_live_experts_per_layer[layer];
                if (quota < 0 || quota > input.num_experts ||
                    quota > unassigned[layer])
                {
                    throw std::invalid_argument(
                        "ExpertOverlay fixed tier quotas over-assign a model layer");
                }
                for (int copy = 0; copy < quota; ++copy)
                {
                    if (!addOneLiveExpert(
                            tier,
                            static_cast<int>(layer),
                            footprints,
                            resource_index,
                            resources))
                    {
                        throwTierCapacityFailure(
                            tier,
                            static_cast<int>(layer),
                            quota,
                            "fixed live quota",
                            footprints,
                            resource_index,
                            resources);
                    }
                }
                unassigned[layer] -= quota;
            }
        }

        if (input.initial_residency_policy ==
            MoEOverlayInitialResidencyPolicy::
                MigrationSourcePerParticipant)
        {
            /*
             * Economy certification measures every directed edge with a real
             * closed transfer cycle. Reserve only the topology-derived source
             * minimum before ordinary priority fill so no declared endpoint is
             * empty at certification time. This is initial data, not durable
             * topology: the published runtime authority may later migrate the
             * final source away when doing so is economical.
             */
            for (auto &tier : tiers)
            {
                const std::size_t participant_count =
                    tier.request->participants.size();
                if (tier.request->copy_policy ==
                        MoEOverlayTierCopyPolicy::Apportioned &&
                    participant_count >
                        static_cast<std::size_t>(input.num_experts))
                {
                    std::ostringstream error;
                    error << "ExpertOverlay tier '"
                          << tier.request->tier_name
                          << "' has " << participant_count
                          << " migration participants but the model has only "
                          << input.num_experts
                          << " experts per layer";
                    throw std::invalid_argument(error.str());
                }
                const int source_quota =
                    tier.request->copy_policy ==
                            MoEOverlayTierCopyPolicy::Replicated
                        ? 1
                        : static_cast<int>(participant_count);

                for (std::size_t layer = 0; layer < layer_count; ++layer)
                {
                    if (tier.request->quota_mode ==
                        MoEOverlayLiveQuotaMode::FixedPerLayer)
                    {
                        if (tier.output.live_experts_per_layer[layer] <
                            source_quota)
                        {
                            std::ostringstream error;
                            error << "ExpertOverlay fixed tier '"
                                  << tier.request->tier_name
                                  << "' provides "
                                  << tier.output.live_experts_per_layer[layer]
                                  << " experts for layer " << layer
                                  << " but migration certification requires "
                                  << source_quota
                                  << " to seed every participant";
                            throw std::invalid_argument(error.str());
                        }
                        continue;
                    }

                    while (tier.output.live_experts_per_layer[layer] <
                           source_quota)
                    {
                        if (unassigned[layer] == 0)
                        {
                            std::ostringstream error;
                            error << "ExpertOverlay cannot seed every migration "
                                     "participant for tier '"
                                  << tier.request->tier_name << "' layer "
                                  << layer << ": all " << input.num_experts
                                  << " model experts are already assigned";
                            throw std::invalid_argument(error.str());
                        }
                        if (!addOneLiveExpert(
                                tier,
                                static_cast<int>(layer),
                                footprints,
                                resource_index,
                                resources))
                        {
                            throwTierCapacityFailure(
                                tier,
                                static_cast<int>(layer),
                                source_quota,
                                "initial migration-source seed",
                                footprints,
                                resource_index,
                                resources);
                        }
                        --unassigned[layer];
                    }
                }
            }
        }

        /* Automatic non-coverage tiers consume balanced slots by priority. */
        for (auto &tier : tiers)
        {
            if (tier.request->fallback ||
                tier.request->quota_mode !=
                    MoEOverlayLiveQuotaMode::Automatic)
            {
                continue;
            }

            while (true)
            {
                std::vector<int> candidates;
                for (std::size_t layer = 0; layer < layer_count; ++layer)
                {
                    if (unassigned[layer] > 0)
                        candidates.push_back(static_cast<int>(layer));
                }
                std::sort(candidates.begin(), candidates.end(), [&](int lhs, int rhs)
                {
                    const int left = tier.output.live_experts_per_layer[
                        static_cast<std::size_t>(lhs)];
                    const int right = tier.output.live_experts_per_layer[
                        static_cast<std::size_t>(rhs)];
                    return left != right ? left < right : lhs < rhs;
                });

                bool admitted = false;
                for (const int layer : candidates)
                {
                    if (addOneLiveExpert(
                            tier, layer, footprints,
                            resource_index, resources))
                    {
                        --unassigned[static_cast<std::size_t>(layer)];
                        admitted = true;
                        break;
                    }
                }
                if (!admitted)
                    break;
            }
        }

        auto fallback = std::find_if(
            tiers.begin(), tiers.end(), [](const auto &tier)
            { return tier.request->fallback; });
        if (fallback == tiers.end())
            throw std::logic_error("Validated ExpertOverlay fallback disappeared");
        if (fallback->request->quota_mode ==
            MoEOverlayLiveQuotaMode::Automatic)
        {
            for (std::size_t layer = 0; layer < layer_count; ++layer)
            {
                const int remainder = unassigned[layer];
                for (int copy = 0; copy < remainder; ++copy)
                {
                    if (!addOneLiveExpert(
                            *fallback,
                            static_cast<int>(layer),
                            footprints,
                            resource_index,
                            resources))
                    {
                        throwTierCapacityFailure(
                            *fallback,
                            static_cast<int>(layer),
                            remainder,
                            "fallback remainder",
                            footprints,
                            resource_index,
                            resources);
                    }
                }
                unassigned[layer] = 0;
            }
        }

        if (std::any_of(unassigned.begin(), unassigned.end(),
                        [](int count) { return count != 0; }))
        {
            throw std::invalid_argument(
                "ExpertOverlay fixed fallback quota leaves uncovered experts after automatic priority fill");
        }

        MoEOverlayResolvedCapacityPlan result;
        result.num_experts = input.num_experts;
        result.layer_footprints = footprints;
        result.tiers.reserve(tiers.size());
        for (auto &tier : tiers)
            result.tiers.push_back(std::move(tier.output));
        std::sort(result.tiers.begin(), result.tiers.end(),
                  [](const auto &lhs, const auto &rhs)
                  { return lhs.tier_index < rhs.tier_index; });

        PhysicalMemoryPlanBuilder physical_plan_builder;
        for (const auto &resource : resources)
        {
            auto final_bom = resource.memory.build();
            if (!final_bom.fits())
            {
                throw std::logic_error(
                    "ExpertOverlay capacity resolver committed an over-budget resource");
            }
            physical_plan_builder.add(final_bom);
        }
        result.physical_memory_admission = std::make_shared<
            const PhysicalMemoryPlanAdmissionCertificate>(
            physical_plan_builder.build());

        result.physical_resources.reserve(resources.size());
        for (auto &resource : resources)
        {
            const auto final_bom = resource.memory.build();
            const PhysicalMemoryAllocatorIdentity identity{
                .world_rank = final_bom.resource().world_rank,
                .device = final_bom.resource().device,
            };
            result.physical_resources.emplace_back(
                std::move(resource.resource_id),
                result.physical_memory_admission,
                identity,
                std::move(resource.live_copies_per_layer),
                std::move(resource.shadow_arrival_capacity_per_layer));
        }
        return result;
    }

    MoERoutedExpertPlacementPlan
    MoEOverlayCapacityResolver::installResolvedQuotas(
        const MoERoutedExpertPlacementPlan &plan,
        const MoEOverlayResolvedCapacityPlan &capacity)
    {
        if (capacity.num_experts <= 0 ||
            capacity.layer_footprints.empty() ||
            capacity.tiers.size() != plan.routed_tiers.size())
        {
            throw std::invalid_argument(
                "ExpertOverlay resolved capacity does not match placement-plan geometry");
        }

        MoERoutedExpertPlacementPlan result = plan;
        std::vector<bool> installed(result.routed_tiers.size(), false);
        for (const auto &resolved : capacity.tiers)
        {
            if (resolved.tier_index < 0 ||
                static_cast<size_t>(resolved.tier_index) >=
                    result.routed_tiers.size() ||
                installed[static_cast<size_t>(resolved.tier_index)])
            {
                throw std::invalid_argument(
                    "ExpertOverlay resolved capacity has an invalid or duplicate tier index");
            }
            if (resolved.live_experts_per_layer.size() !=
                capacity.layer_footprints.size())
            {
                throw std::invalid_argument(
                    "ExpertOverlay resolved capacity omits model-layer quota geometry");
            }

            auto &tier = result.routed_tiers[
                static_cast<size_t>(resolved.tier_index)];
            if (tier.name != resolved.tier_name ||
                tier.priority != resolved.priority ||
                tier.fallback != resolved.fallback)
            {
                throw std::invalid_argument(
                    "ExpertOverlay resolved capacity tier identity does not match the placement plan");
            }
            if (!tier.resolved_live_experts_per_layer.empty() &&
                tier.resolved_live_experts_per_layer !=
                    resolved.live_experts_per_layer)
            {
                throw std::invalid_argument(
                    "ExpertOverlay installed placement quotas differ from the retained physical-memory certificate");
            }
            tier.resolved_live_experts_per_layer =
                resolved.live_experts_per_layer;
            installed[static_cast<size_t>(resolved.tier_index)] = true;
        }
        if (std::find(installed.begin(), installed.end(), false) !=
            installed.end())
        {
            throw std::invalid_argument(
                "ExpertOverlay resolved capacity omits a placement tier");
        }

        const auto validation = validateMoERoutedExpertPlacementPlan(
            result,
            MoERoutedExpertPlacementValidationOptions{
                .layer_count = static_cast<int>(
                    capacity.layer_footprints.size()),
                .routed_expert_count = capacity.num_experts,
            });
        if (!validation.ok())
        {
            std::ostringstream message;
            message << "ExpertOverlay resolved capacity produced an invalid placement plan:";
            for (const auto &error : validation.errors)
                message << "\n - " << error;
            throw std::invalid_argument(message.str());
        }
        return result;
    }
} // namespace llaminar2
