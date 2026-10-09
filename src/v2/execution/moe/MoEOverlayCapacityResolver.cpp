/**
 * @file MoEOverlayCapacityResolver.cpp
 * @brief Exact prepared-weight BOM and integer-priority quota resolution.
 *
 * All arithmetic in this file is setup-time and fail-closed.  The implementation
 * mirrors the allocation layouts used by CpuExpertSlotPool and
 * GpuExpertSlotPool, then admits logical residents one at a time while charging
 * their actual physical copy multiplicity.  No runtime allocator or backend is
 * consulted, which keeps the same decision reproducible on every MPI rank.
 * Pipeline stages share the same typed physical BOM throughout admission;
 * compact stage quota/count metadata never becomes an independent byte budget.
 */

#include "MoEOverlayCapacityResolver.h"

#include "ExpertPreparedMemoryGeometry.h"
#include "MoEExpertProjectionOwnership.h"

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

        /** @brief One setup-only physical BOM shared by all pipeline stages. */
        struct ResourceState
        {
            PhysicalMemoryBOMBuilder memory;
            /**
             * @brief Extend one canonical fixed BOM during setup admission.
             * @param base Aggregate stage contributions for this allocator.
             */
            explicit ResourceState(const PhysicalMemoryBOM &base)
                : memory(base)
            {
            }
        };

        /** @brief Immutable per-tier byte geometry beside its setup quota candidate. */
        struct TierState
        {
            const MoEOverlayTierCapacityRequest *request = nullptr;
            MoEOverlayResolvedTierCapacity output;
            std::vector<MoEOverlayPreparedExpertFootprint> movable_footprints;
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

        /**
         * @brief Price the immutable down bank independently of movable owner counts.
         * @param input Complete model manifest and frozen tier copy policies.
         * @param resource_index Validated participant-to-physical-resource join.
         * @param resources Canonical base BOMs before expert residency is added.
         * @return Unconditional routed-weight contributions for each physical resource.
         * @throws std::invalid_argument for an unsupported topology or source layout.
         *
         * The first installed projection pipeline covers a complete homogeneous
         * GPU domain. A tier containing only some experts cannot claim the same
         * bank: cross-tier slice arrival needs its own publication protocol. Fail
         * here instead of admitting a cheap but physically incomplete layout.
         */
        [[nodiscard]] std::map<std::string, std::size_t> fixedDownCharges(
            const MoEOverlayCapacityResolverInput &input,
            const std::unordered_map<std::string, std::size_t> &resource_index,
            const std::vector<ResourceState> &resources)
        {
            std::map<std::string, std::size_t> charges;
            for (const auto &tier : input.tiers)
            {
                if (tier.copy_policy != MoEOverlayTierCopyPolicy::GateUpOwnedDownColumns)
                    continue;
                if (input.tiers.size() != 1u || tier.participants.size() >
                    static_cast<std::size_t>(std::numeric_limits<int>::max()))
                    throw std::invalid_argument("Gate/up capacity requires one complete GPU domain");
                const auto first_device = resources.at(resource_index.at(
                    tier.participants.front().resource_id)).memory.build().resource().device;
                std::set<std::string> unique_resources;
                for (std::size_t index = 0; index < tier.participants.size(); ++index)
                {
                    const auto &participant = tier.participants[index];
                    const auto device = resources.at(resource_index.at(
                        participant.resource_id)).memory.build().resource().device;
                    if (!device.is_gpu() || device.type != first_device.type ||
                        !unique_resources.insert(participant.resource_id).second ||
                        participant.shadow_slots_per_layer != 0u)
                        throw std::invalid_argument(
                            "Gate/up capacity requires distinct homogeneous GPUs and native transfer-directory storage");

                    for (const auto &layer : input.layer_weight_manifest)
                    {
                        // valid() has already proved every role occurs once.
                        // Address projections by role, never by array position.
                        const auto &gate = *std::find_if(layer.projections.begin(), layer.projections.end(),
                            [](const auto &p) { return p.projection == ExpertTierWeightProjection::Gate; });
                        const auto &up = *std::find_if(layer.projections.begin(), layer.projections.end(),
                            [](const auto &p) { return p.projection == ExpertTierWeightProjection::Up; });
                        const auto &down = *std::find_if(layer.projections.begin(), layer.projections.end(),
                            [](const auto &p) { return p.projection == ExpertTierWeightProjection::Down; });
                        if (gate.N != up.N || gate.K != up.K || down.N != gate.K || down.K != gate.N)
                            throw std::invalid_argument("Gate/up capacity has incompatible source projection dimensions");
                        const auto ownership = MoEExpertProjectionOwnership::gateUpOwnedDownColumns(
                            {input.num_experts, gate.K, gate.N}, static_cast<int>(index),
                            static_cast<int>(tier.participants.size()));
                        const auto slice = ownership.projection(WeightRole::MoEExpertDown);
                        const auto geometry = resolveExpertPreparedProjectionMemoryGeometry(
                            slice.rows, slice.source_columns, down.format);
                        const auto bytes = checkedMultiply(
                            ownership.residentExperts(WeightRole::MoEExpertDown, 0),
                            geometry.gpu_live_bytes, "fixed down projection bank");
                        charges[participant.resource_id] = checkedAdd(
                            charges[participant.resource_id], bytes, "fixed down layer inventory");
                    }
                }
            }
            return charges;
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
            const std::vector<MoEOverlayLayerWeightManifest> &manifest,
            const std::unordered_map<std::string, std::size_t> &resource_index,
            const std::vector<ResourceState> &resources)
        {
            const auto &footprints = tier.movable_footprints;
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
            const std::unordered_map<std::string, std::size_t> &resource_index,
            const std::vector<ResourceState> &resources)
        {
            const auto &footprints = tier.movable_footprints;
            std::map<std::string, std::size_t> charges;
            const int quota =
                tier.output.live_experts_per_layer[static_cast<std::size_t>(layer_idx)];
            if (tier.request->copy_policy != MoEOverlayTierCopyPolicy::Replicated)
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
                tier, layer_idx, resource_index, resources);
            if (!chargesFit(live_charges, resource_index, resources))
                return false;
            commitCharges(
                live_charges, resource_index, resources, /*shadow=*/false);

            const int previous_quota = tier.output.live_experts_per_layer[layer];
            ++tier.output.live_experts_per_layer[layer];
            tier.output.has_live_residency = true;
            if (tier.request->copy_policy != MoEOverlayTierCopyPolicy::Replicated)
            {
                const std::size_t participant =
                    static_cast<std::size_t>(previous_quota) %
                    tier.request->participants.size();
                ++tier.output.participant_live_copies[participant][layer];

            }
            else
            {
                for (std::size_t participant = 0;
                     participant < tier.request->participants.size();
                     ++participant)
                {
                    ++tier.output.participant_live_copies[participant][layer];

                }
            }
            return true;
        }

        /**
         * @brief Reject an unmet quota with the exact limiting physical owners.
         * @param tier Logical tier requesting another complete expert.
         * @param layer_idx Layer whose next allocation could not be admitted.
         * @param requested_quota Required logical expert count for this demand.
         * @param demand Diagnostic name of the mandatory coverage obligation.
         * @param resource_index Logical participant to physical resource lookup.
         * @param resources Current setup BOMs, never a separate runtime ledger.
         * @throws std::invalid_argument with category-level admission evidence.
         */
        [[noreturn]] void throwTierCapacityFailure(
            const TierState &tier,
            int layer_idx,
            int requested_quota,
            const char *demand,
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
                      // Expose the canonical owners separately. A transfer
                      // directory, weight-loader peak, and activation wire
                      // buffer have different lifetimes and tuning controls;
                      // their aggregate alone cannot identify an admission bug.
                      << " weight_load_staging_bytes="
                      << bom.bytes(PhysicalMemoryOwner::WeightLoadStaging)
                      << " activation_transport_staging_bytes="
                      << bom.bytes(PhysicalMemoryOwner::ActivationTransportStaging)
                      << " expert_migration_staging_bytes="
                      << bom.bytes(PhysicalMemoryOwner::ExpertMigrationStaging)
                      << " shadow_bytes="
                      << bom.bytes(PhysicalMemoryOwner::ExpertShadowSlots)
                      << " live_expert_bytes="
                      << bom.bytes(PhysicalMemoryOwner::RoutedExpertWeights);
            }
            throw MoEOverlayCapacityExhausted(error.str());
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
        const std::vector<MoEOverlayLayerWeightManifest> &manifest,
        DeviceMoEProjectionSet projections)
    {
        if (manifest.empty() || !deviceMoEProjectionSetValid(projections))
            throw std::invalid_argument("ExpertOverlay capacity resolver requires a layer manifest and a valid projection family");

        const int first_model_layer = manifest.front().layer_idx;
        if (first_model_layer < 0 ||
            manifest.size() > static_cast<std::size_t>(
                std::numeric_limits<int>::max() - first_model_layer))
            throw std::invalid_argument("ExpertOverlay capacity layer interval is invalid or overflows");

        std::vector<MoEOverlayPreparedExpertFootprint> footprints;
        footprints.reserve(manifest.size());
        for (std::size_t layer_offset = 0;
             layer_offset < manifest.size();
             ++layer_offset)
        {
            const auto &layer = manifest[layer_offset];
            if (!layer.valid() ||
                layer.layer_idx != first_model_layer + static_cast<int>(layer_offset))
            {
                throw std::invalid_argument(
                    "ExpertOverlay capacity layer manifest must be valid, unique, and contiguous within its owned interval");
            }

            MoEOverlayPreparedExpertFootprint footprint;
            footprint.layer_idx = layer.layer_idx;
            for (const auto &projection : layer.projections)
            {
                // Only ownership-dependent bytes scale with the live quota.
                // The complete manifest is still authenticated above, and the
                // fixed down bank is priced before admitting any live owner.
                if (projections == DeviceMoEProjectionSet::GateUp &&
                    projection.projection == ExpertTierWeightProjection::Down)
                    continue;
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

    namespace
    {
        /** @brief Stage-local geometry and quotas; physical bytes have one shared BOM. */
        struct StageState
        {
            const MoEOverlayCapacityResolverInput *input;
            std::vector<MoEOverlayPreparedExpertFootprint> footprints;
            std::vector<TierState> tiers;
            std::vector<int> unassigned;
        };

        /**
         * @brief Validate a complete stage before any capacity search can reject it.
         * @param input Stage-owned manifest, resource contributions and tier policies.
         * @param resource_index Complete physical-resource namespace of the pipeline.
         * @return Compact stage quota state with no private memory budget.
         * @throws std::invalid_argument for malformed geometry, quotas or bindings.
         * @throws std::overflow_error for unrepresentable expert footprints.
         */
        [[nodiscard]] StageState prepareStage(
            const MoEOverlayCapacityResolverInput &input,
            const std::unordered_map<std::string, std::size_t> &resource_index)
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
            auto footprints = MoEOverlayCapacityResolver::preparedFootprints(
                input.layer_weight_manifest);
            const std::size_t layer_count = footprints.size();
            if (input.physical_budgets.empty() || input.tiers.empty())
            {
                throw std::invalid_argument(
                    "ExpertOverlay capacity requires physical budgets and routed tiers");
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
                        std::none_of(input.physical_budgets.begin(), input.physical_budgets.end(),
                            [&](const auto &budget) { return budget.resourceId() == participant.resource_id; }) ||
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
                state.movable_footprints = MoEOverlayCapacityResolver::preparedFootprints(
                    input.layer_weight_manifest, moeOverlayMovableProjections(request.copy_policy));
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

            return {&input, std::move(footprints), std::move(tiers),
                    std::vector<int>(layer_count, input.num_experts)};
        }

        /**
         * @brief Charge all projection and shadow banks before automatic placement.
         * @param stage Validated compact stage quotas and immutable geometry.
         * @param resource_index Diagnostic identities joined to physical allocators.
         * @param resources The one setup BOM shared by every pipeline stage.
         * @throws MoEOverlayCapacityExhausted when required owners cannot fit.
         * @throws std::invalid_argument for unsatisfiable exact quota constraints.
         * @throws std::overflow_error when a typed charge cannot be represented.
         */
        void chargeFixedBanks(StageState &stage,
            const std::unordered_map<std::string, std::size_t> &resource_index,
            std::vector<ResourceState> &resources)
        {
            const auto &input = *stage.input;
            auto &tiers = stage.tiers;
            // Fixed down slices cover every expert, including on a participant
            // whose movable quota is zero. Add them to the same physical BOM before
            // quota search; no second ledger or post-admission discount is involved.
            const auto fixed_down = fixedDownCharges(input, resource_index, resources);
            if (!chargesFit(fixed_down, resource_index, resources))
                throw MoEOverlayCapacityExhausted("ExpertOverlay fixed down banks exceed the complete physical BOM");
            commitCharges(fixed_down, resource_index, resources, /*shadow=*/false);

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
                std::ostringstream error;
                error << "ExpertOverlay endpoint shadow banks exceed the complete physical BOM";
                for (const auto &[resource_id, shadow_bytes] : all_shadow_charges)
                {
                    const auto &resource = resources[resource_index.at(resource_id)];
                    const auto bom = resource.memory.build();
                    const auto used = bom.incrementalBytes();
                    const auto available = bom.resource().admission_available_bytes;
                    if (used <= available && shadow_bytes <= available - used)
                        continue;
                    error << "; resource='" << resource_id << "' device="
                          << bom.resource().device.toString()
                          << " fixed_bytes=" << used
                          << " shadow_bytes=" << shadow_bytes
                          << " usable_bytes=" << available;
                }
                throw MoEOverlayCapacityExhausted(error.str());
            }
            commitCharges(
                all_shadow_charges, resource_index, resources, /*shadow=*/true);
        }

        /**
         * @brief Reserve exact quotas and migration sources across every stage.
         * @param stage Validated compact stage quotas and immutable geometry.
         * @param resource_index Diagnostic identities joined to physical allocators.
         * @param resources The one setup BOM shared by every pipeline stage.
         * @throws MoEOverlayCapacityExhausted when required owners cannot fit.
         * @throws std::invalid_argument for unsatisfiable exact quota constraints.
         * @throws std::overflow_error when a typed charge cannot be represented.
         */
        void admitRequiredQuotas(StageState &stage,
            const std::unordered_map<std::string, std::size_t> &resource_index,
            std::vector<ResourceState> &resources)
        {
            const auto &input = *stage.input;
            auto &tiers = stage.tiers;
            const auto layer_count = stage.footprints.size();
            auto &unassigned = stage.unassigned;
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
                                resource_index,
                                resources))
                        {
                            throwTierCapacityFailure(
                                tier,
                                static_cast<int>(layer),
                                quota,
                                "fixed live quota",
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
                    if (tier.request->copy_policy !=
                            MoEOverlayTierCopyPolicy::Replicated &&
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
                                    resource_index,
                                    resources))
                            {
                                throwTierCapacityFailure(
                                    tier,
                                    static_cast<int>(layer),
                                    source_quota,
                                    "initial migration-source seed",
                                    resource_index,
                                    resources);
                            }
                            --unassigned[layer];
                        }
                    }
                }
            }

        }

        /**
         * @brief Fill this stage's preferred tiers in its authored priority order.
         * @param stage Validated compact stage quotas and immutable geometry.
         * @param resource_index Diagnostic identities joined to physical allocators.
         * @param resources The one setup BOM shared by every pipeline stage.
         * @throws MoEOverlayCapacityExhausted when required owners cannot fit.
         * @throws std::invalid_argument for unsatisfiable exact quota constraints.
         * @throws std::overflow_error when a typed charge cannot be represented.
         */
        void admitPreferredQuotas(StageState &stage,
            const std::unordered_map<std::string, std::size_t> &resource_index,
            std::vector<ResourceState> &resources)
        {
            auto &tiers = stage.tiers;
            const auto layer_count = stage.footprints.size();
            auto &unassigned = stage.unassigned;
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
                                tier, layer,
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

        }

        /**
         * @brief Complete this stage's expert coverage against the shared BOM.
         * @param stage Validated compact stage quotas and immutable geometry.
         * @param resource_index Diagnostic identities joined to physical allocators.
         * @param resources The one setup BOM shared by every pipeline stage.
         * @throws MoEOverlayCapacityExhausted when required owners cannot fit.
         * @throws std::invalid_argument for unsatisfiable exact quota constraints.
         * @throws std::overflow_error when a typed charge cannot be represented.
         */
        void admitCoverage(StageState &stage,
            const std::unordered_map<std::string, std::size_t> &resource_index,
            std::vector<ResourceState> &resources)
        {
            auto &tiers = stage.tiers;
            const auto layer_count = stage.footprints.size();
            auto &unassigned = stage.unassigned;
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
                                resource_index,
                                resources))
                        {
                            throwTierCapacityFailure(
                                *fallback,
                                static_cast<int>(layer),
                                remainder,
                                "fallback remainder",
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

        }

        /**
         * @brief Publish stage-local counts beside the sole aggregate byte authority.
         * @param stage Fully covered stage with exact compact participant quotas.
         * @param authority Completed admission of all fixed and routed owners.
         * @return Stage quota plan whose resource views retain that same certificate.
         * @throws std::overflow_error if a diagnostic copy count exceeds int.
         */
        [[nodiscard]] MoEOverlayResolvedCapacityPlan finishStage(
            StageState &&stage,
            const std::shared_ptr<const PhysicalMemoryPlanAdmissionCertificate> &authority)
        {
            MoEOverlayResolvedCapacityPlan result;
            result.num_experts = stage.input->num_experts;
            result.layer_footprints = std::move(stage.footprints);
            result.physical_memory_admission = authority;
            const auto layer_count = result.layer_footprints.size();
            for (const auto &budget : stage.input->physical_budgets)
            {
                std::vector<int> live(layer_count, 0);
                std::vector<int> shadows(layer_count, 0);
                for (const auto &tier : stage.tiers)
                    for (std::size_t participant = 0; participant < tier.request->participants.size(); ++participant)
                    {
                        const auto &endpoint = tier.request->participants[participant];
                        if (endpoint.resource_id != budget.resourceId()) continue;
                        for (std::size_t layer = 0; layer < layer_count; ++layer)
                        {
                            const auto copies = tier.output.participant_live_copies[participant][layer];
                            if (copies > std::numeric_limits<int>::max() - live[layer] ||
                                endpoint.shadow_slots_per_layer > static_cast<std::size_t>(
                                    std::numeric_limits<int>::max() - shadows[layer]))
                                throw std::overflow_error("ExpertOverlay stage copy count exceeds int");
                            live[layer] += copies;
                            shadows[layer] += static_cast<int>(endpoint.shadow_slots_per_layer);
                        }
                    }
                const auto &resource = budget.certificate().bom().resource();
                result.physical_resources.emplace_back(budget.resourceId(), authority,
                    PhysicalMemoryAllocatorIdentity{resource.world_rank, resource.device},
                    std::move(live), std::move(shadows));
            }
            for (auto &tier : stage.tiers) result.tiers.push_back(std::move(tier.output));
            std::sort(result.tiers.begin(), result.tiers.end(),
                [](const auto &left, const auto &right) { return left.tier_index < right.tier_index; });
            return result;
        }
    } // namespace

    MoEOverlayResolvedCapacityPlan MoEOverlayCapacityResolver::resolve(
        const MoEOverlayCapacityResolverInput &input)
    {
        auto resolved = resolvePipeline(std::span(&input, 1u));
        return std::move(resolved.front());
    }

    std::vector<MoEOverlayResolvedCapacityPlan> MoEOverlayCapacityResolver::resolvePipeline(
        std::span<const MoEOverlayCapacityResolverInput> inputs)
    {
        if (inputs.empty())
            throw std::invalid_argument("ExpertOverlay pipeline capacity requires at least one stage");

        // Every contribution joins by allocator identity. The name is only its
        // authenticated protocol label; aliases must not create extra capacity.
        PhysicalMemoryPlanBuilder base_builder;
        std::vector<std::pair<std::string, PhysicalMemoryAllocatorIdentity>> bindings;
        std::unordered_map<std::string, std::size_t> resource_index;
        for (const auto &input : inputs)
        {
            std::set<std::string> stage_resources;
            for (const auto &budget : input.physical_budgets)
            {
                const auto &bom = budget.certificate().bom();
                const auto &physical = bom.resource();
                const PhysicalMemoryAllocatorIdentity identity{physical.world_rank, physical.device};
                if ((!physical.device.is_cpu() && !physical.device.is_gpu()) ||
                    !physical.device.is_valid() || physical.admission_available_bytes == 0u ||
                    !stage_resources.insert(budget.resourceId()).second)
                    throw std::invalid_argument("ExpertOverlay stage requires unique valid physical budgets");
                const auto named = resource_index.find(budget.resourceId());
                if (named == resource_index.end())
                {
                    if (std::any_of(bindings.begin(), bindings.end(),
                        [&](const auto &binding) { return binding.second == identity; }))
                        throw std::invalid_argument("ExpertOverlay physical allocator has multiple resource identities");
                    resource_index.emplace(budget.resourceId(), bindings.size());
                    bindings.emplace_back(budget.resourceId(), identity);
                }
                else if (bindings[named->second].second != identity)
                    throw std::invalid_argument("ExpertOverlay resource identity names different physical allocators");
                // The canonical builder verifies a common capacity observation
                // and accumulates typed owners; never subtract stage budgets.
                base_builder.add(bom);
            }
        }
        const auto base = base_builder.build();
        std::vector<ResourceState> resources;
        resources.reserve(bindings.size());
        for (const auto &binding : bindings)
            resources.emplace_back(*base.find(binding.second));

        std::vector<StageState> stages;
        stages.reserve(inputs.size());
        int previous_last_layer = -1;
        for (const auto &input : inputs)
        {
            auto stage = prepareStage(input, resource_index);
            if (stage.footprints.front().layer_idx <= previous_last_layer)
                throw std::invalid_argument("ExpertOverlay pipeline stages require ordered disjoint layer intervals");
            previous_last_layer = stage.footprints.back().layer_idx;
            stages.push_back(std::move(stage));
        }
        if (!base.fits())
            throw MoEOverlayCapacityExhausted("ExpertOverlay pipeline fixed owners exceed the complete physical BOM: " + base.summary());

        // Fixed obligations of every stage precede optional priority fill. A
        // preceding stage cannot spend a later stage's fixed or migration bank.
        for (auto &stage : stages) chargeFixedBanks(stage, resource_index, resources);
        for (auto &stage : stages) admitRequiredQuotas(stage, resource_index, resources);
        for (auto &stage : stages) admitPreferredQuotas(stage, resource_index, resources);
        for (auto &stage : stages) admitCoverage(stage, resource_index, resources);

        PhysicalMemoryPlanBuilder completed;
        for (const auto &resource : resources) completed.add(resource.memory.build());
        const auto authority = std::make_shared<const PhysicalMemoryPlanAdmissionCertificate>(completed.build());
        std::vector<MoEOverlayResolvedCapacityPlan> results;
        results.reserve(stages.size());
        for (auto &stage : stages) results.push_back(finishStage(std::move(stage), authority));
        return results;
    }

    MoERoutedExpertPlacementPlan
    MoEOverlayCapacityResolver::installResolvedQuotas(
        const MoERoutedExpertPlacementPlan &plan,
        const MoEOverlayResolvedCapacityPlan &capacity)
    {
        if (capacity.num_experts <= 0 ||
            capacity.layer_footprints.empty() ||
            plan.first_model_layer < 0 ||
            capacity.layer_footprints.size() > static_cast<std::size_t>(
                std::numeric_limits<int>::max() - plan.first_model_layer) ||
            capacity.tiers.size() != plan.routed_tiers.size())
        {
            throw std::invalid_argument(
                "ExpertOverlay resolved capacity does not match placement-plan geometry");
        }

        // Equal row counts do not establish stage identity. The footprint
        // manifest is the authority for global layer IDs; quota arrays retain
        // only the compact owned interval, never preceding-stage padding.
        for (std::size_t row = 0; row < capacity.layer_footprints.size(); ++row)
        {
            if (capacity.layer_footprints[row].layer_idx !=
                plan.first_model_layer + static_cast<int>(row))
                throw std::invalid_argument(
                    "ExpertOverlay resolved capacity belongs to a different placement-plan layer interval");
        }

        MoERoutedExpertPlacementPlan result = plan;
        if (plan.replica_cache_capacity &&
            plan.replica_cache_capacity != capacity.replica_cache_capacity)
            throw std::logic_error("ExpertOverlay retained replica-cache grant changed during admission");
        result.replica_cache_capacity = capacity.replica_cache_capacity;
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
