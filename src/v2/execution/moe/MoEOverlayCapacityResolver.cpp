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

#include "kernels/cpu/gemm/CPUNativeVNNIWeightPacker.h"
#include "tensors/NativeVnniFormatInfo.h"

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
        constexpr std::size_t kGpuRegionAlignment = 256;
        constexpr std::size_t kCpuAllocationAlignment = 4096;

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

        [[nodiscard]] std::size_t alignUp(
            std::size_t bytes,
            std::size_t alignment,
            const char *what)
        {
            if (alignment == 0 || (alignment & (alignment - 1)) != 0)
                throw std::logic_error("ExpertOverlay capacity alignment must be a power of two");
            return checkedAdd(bytes, alignment - 1, what) & ~(alignment - 1);
        }

        [[nodiscard]] std::size_t gpuProjectionAllocationBytes(
            int N,
            int K,
            int payload_bytes_per_block,
            bool is_asymmetric,
            bool has_emins)
        {
            if (N <= 0 || K <= 0 || K % 32 != 0 ||
                payload_bytes_per_block <= 0)
            {
                throw std::invalid_argument(
                    "ExpertOverlay GPU footprint requires positive 32-aligned geometry");
            }

            const std::size_t blocks = checkedMultiply(
                static_cast<std::size_t>(N),
                static_cast<std::size_t>(K / 32),
                "GPU block count");
            std::size_t cursor = 0;
            const auto append_region = [&](std::size_t bytes, const char *what)
            {
                if (bytes == 0)
                    return;
                cursor = alignUp(cursor, kGpuRegionAlignment, what);
                cursor = checkedAdd(cursor, bytes, what);
            };

            append_region(
                checkedMultiply(
                    blocks,
                    static_cast<std::size_t>(payload_bytes_per_block),
                    "GPU payload"),
                "GPU payload");
            append_region(
                checkedMultiply(blocks, sizeof(std::uint16_t), "GPU scales"),
                "GPU scales");
            if (is_asymmetric)
            {
                append_region(
                    checkedMultiply(blocks, sizeof(std::uint16_t), "GPU minima"),
                    "GPU minima");
            }
            if (has_emins)
            {
                append_region(
                    checkedMultiply(blocks, sizeof(std::uint32_t), "GPU effective minima"),
                    "GPU effective minima");
            }
            return alignUp(cursor, kGpuRegionAlignment, "GPU projection allocation");
        }

        [[nodiscard]] std::size_t cpuProjectionAllocationBytes(
            const MoEOverlayProjectionWeightManifest &projection,
            const NativeVnniFormatInfo &source)
        {
            const auto encoding =
                cpu::native_vnni::preparedEncodingForCodebook(
                    source.codebook_id);
            const std::size_t stride =
                cpu::native_vnni::preparedInterleavedBlockStride(
                    encoding, source.is_asymmetric);
            const std::size_t padded_n =
                alignUp(static_cast<std::size_t>(projection.N), 64, "CPU padded N");
            const std::size_t units = checkedMultiply(
                padded_n / 64,
                static_cast<std::size_t>(projection.K / 32),
                "CPU NativeVNNI unit count");
            const std::size_t logical_bytes = checkedMultiply(
                units, stride, "CPU NativeVNNI projection");

            /*
             * CpuExpertSlotPool owns every projection in a separate
             * AlignedVector. Large vectors are page aligned and page rounded;
             * NUMA-bound ranges explicitly require the same 4 KiB contract.
             */
            return alignUp(
                logical_bytes,
                logical_bytes >= kCpuAllocationAlignment
                    ? kCpuAllocationAlignment
                    : std::size_t{64},
                "CPU projection allocation");
        }

        /** @brief Charge one contiguous floating projection with pool alignment. */
        [[nodiscard]] std::size_t floatingProjectionAllocationBytes(
            const MoEOverlayProjectionWeightManifest &projection,
            std::size_t alignment,
            const char *description)
        {
            const std::size_t elements = checkedMultiply(
                static_cast<std::size_t>(projection.N),
                static_cast<std::size_t>(projection.K),
                description);
            const std::size_t logical_bytes = checkedMultiply(
                elements,
                projection.format.floatingElementBytes(),
                description);
            return alignUp(logical_bytes, alignment, description);
        }

        struct ResourceState
        {
            MoEOverlayResolvedPhysicalMemory output;
        };

        struct TierState
        {
            const MoEOverlayTierCapacityRequest *request = nullptr;
            MoEOverlayResolvedTierCapacity output;
        };

        [[nodiscard]] std::size_t resourceUsed(
            const MoEOverlayResolvedPhysicalMemory &resource)
        {
            std::size_t used = 0;
            used = checkedAdd(used, resource.fixed_bytes, "physical fixed");
            used = checkedAdd(
                used, resource.transfer_staging_bytes, "physical staging");
            used = checkedAdd(
                used, resource.safety_reserve_bytes, "physical reserve");
            used = checkedAdd(used, resource.shadow_bytes, "physical shadows");
            used = checkedAdd(
                used, resource.live_expert_bytes, "physical live experts");
            return used;
        }

        [[nodiscard]] bool resourceCanAdd(
            const ResourceState &resource,
            std::size_t bytes)
        {
            const std::size_t used = resourceUsed(resource.output);
            return used <= resource.output.usable_budget_bytes &&
                   bytes <= resource.output.usable_budget_bytes - used;
        }

        [[nodiscard]] std::size_t footprintFor(
            const MoEOverlayPreparedExpertFootprint &footprint,
            DeviceId device,
            bool shadow)
        {
            return shadow ? footprint.shadowBytes(device)
                          : footprint.liveBytes(device);
        }

        [[nodiscard]] std::map<std::string, std::size_t> tierShadowCharges(
            const TierState &tier,
            const std::vector<MoEOverlayPreparedExpertFootprint> &footprints,
            const std::unordered_map<std::string, std::size_t> &resource_index,
            const std::vector<ResourceState> &resources)
        {
            std::map<std::string, std::size_t> charges;
            for (const auto &participant : tier.request->participants)
            {
                const auto found = resource_index.find(participant.resource_id);
                if (found == resource_index.end())
                    throw std::logic_error("Validated ExpertOverlay resource disappeared");
                const DeviceId device = resources[found->second].output.device;
                for (const auto &footprint : footprints)
                {
                    const std::size_t bytes = checkedMultiply(
                        participant.shadow_slots_per_layer,
                        footprintFor(footprint, device, /*shadow=*/true),
                        "tier shadow slots");
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
                    resources[resource].output.device,
                    /*shadow=*/false);
                return charges;
            }

            for (const auto &participant : tier.request->participants)
            {
                const auto resource = resource_index.at(participant.resource_id);
                const std::size_t bytes = footprintFor(
                    footprints[static_cast<std::size_t>(layer_idx)],
                    resources[resource].output.device,
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
                auto &output = resources[resource_index.at(resource_id)].output;
                if (shadow)
                    output.shadow_bytes = checkedAdd(output.shadow_bytes, bytes, "committed shadows");
                else
                    output.live_expert_bytes = checkedAdd(
                        output.live_expert_bytes, bytes, "committed live experts");
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
                    tier.request->participants[participant].resource_id)].output;
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
                        tier.request->participants[participant].resource_id)].output;
                    ++resource.live_copies_per_layer[layer];
                }
            }
            return true;
        }

        [[noreturn]] void throwTierCapacityFailure(
            const TierState &tier,
            int layer_idx,
            int requested_quota,
            const std::vector<MoEOverlayPreparedExpertFootprint> &footprints,
            const std::unordered_map<std::string, std::size_t> &resource_index,
            const std::vector<ResourceState> &resources)
        {
            std::ostringstream error;
            error << "ExpertOverlay tier '" << tier.request->tier_name
                  << "' cannot admit its "
                  << (tier.request->fallback ? "fallback remainder" : "fixed live quota")
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
             * whether fixed graph state, live experts, or another reserved
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
                    resources[resource_index.at(resource_id)].output;
                const std::size_t used = resourceUsed(resource);
                const std::size_t remaining =
                    used < resource.usable_budget_bytes
                        ? resource.usable_budget_bytes - used
                        : 0;
                error << "; resource='" << resource_id
                      << "' device=" << resource.device.toString()
                      << " requires_additional_bytes=" << additional_bytes
                      << " remaining_bytes=" << remaining
                      << " usable_bytes=" << resource.usable_budget_bytes
                      << " fixed_bytes=" << resource.fixed_bytes
                      << " staging_bytes=" << resource.transfer_staging_bytes
                      << " safety_reserve_bytes="
                      << resource.safety_reserve_bytes
                      << " shadow_bytes=" << resource.shadow_bytes
                      << " live_expert_bytes=" << resource.live_expert_bytes;
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
            { return entry.resource_id == resource_id; });
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
                if (projection.format.isFloating())
                {
                    const std::size_t floating_elements = checkedMultiply(
                        static_cast<std::size_t>(projection.N),
                        static_cast<std::size_t>(projection.K),
                        "floating expert elements");
                    const std::size_t floating_logical_bytes = checkedMultiply(
                        floating_elements,
                        projection.format.floatingElementBytes(),
                        "floating expert bytes");
                    const std::size_t cpu_bytes =
                        floatingProjectionAllocationBytes(
                            projection,
                            floating_logical_bytes >= kCpuAllocationAlignment
                                ? kCpuAllocationAlignment
                                : std::size_t{64},
                            "CPU floating expert projection");
                    const std::size_t gpu_bytes =
                        floatingProjectionAllocationBytes(
                            projection,
                            kGpuRegionAlignment,
                            "GPU floating expert projection");
                    footprint.cpu_live_bytes = checkedAdd(
                        footprint.cpu_live_bytes,
                        cpu_bytes,
                        "CPU floating live expert");
                    footprint.cpu_shadow_bytes = checkedAdd(
                        footprint.cpu_shadow_bytes,
                        cpu_bytes,
                        "CPU floating shadow expert");
                    footprint.gpu_live_bytes = checkedAdd(
                        footprint.gpu_live_bytes,
                        gpu_bytes,
                        "GPU floating live expert");
                    footprint.gpu_shadow_bytes = checkedAdd(
                        footprint.gpu_shadow_bytes,
                        gpu_bytes,
                        "GPU floating shadow expert");
                    continue;
                }

                const NativeVnniFormatInfo *source =
                    native_vnni_formats::forSourceIdentity(
                        projection.format.native_vnni.codebook_id,
                        projection.format.native_vnni.is_superblock);
                if (!source)
                {
                    throw std::invalid_argument(
                        "ExpertOverlay capacity projection source is not in the NativeVNNI catalog");
                }

                const std::size_t cpu_bytes =
                    cpuProjectionAllocationBytes(projection, *source);
                footprint.cpu_live_bytes = checkedAdd(
                    footprint.cpu_live_bytes, cpu_bytes, "CPU live expert");
                footprint.cpu_shadow_bytes = checkedAdd(
                    footprint.cpu_shadow_bytes, cpu_bytes, "CPU shadow expert");

                const auto live_allocation =
                    reusableDeviceVnniAllocationFormat(*source);
                footprint.gpu_live_bytes = checkedAdd(
                    footprint.gpu_live_bytes,
                    gpuProjectionAllocationBytes(
                        projection.N,
                        projection.K,
                        live_allocation.payload_bytes_per_block,
                        live_allocation.has_mins,
                        live_allocation.has_emins),
                    "GPU live expert");

                /*
                 * The installed physical fabric deliberately provisions every
                 * GPU arrival slot for the largest catalogued representation:
                 * signed INT8 payload, minima, and effective minima.  Charging
                 * exactly that allocation prevents codebook-dependent runtime
                 * pool construction from escaping the setup BOM.
                 */
                footprint.gpu_shadow_bytes = checkedAdd(
                    footprint.gpu_shadow_bytes,
                    gpuProjectionAllocationBytes(
                        projection.N,
                        projection.K,
                        /*payload_bytes_per_block=*/32,
                        /*is_asymmetric=*/true,
                        /*has_emins=*/true),
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
            if (budget.resource_id.empty() || !budget.device.is_valid() ||
                (!budget.device.is_cpu() && !budget.device.is_gpu()) ||
                budget.usable_budget_bytes == 0 ||
                !resource_index.emplace(
                     budget.resource_id, resources.size()).second)
            {
                throw std::invalid_argument(
                    "ExpertOverlay physical budgets require unique non-empty identities, valid devices, and positive usable bytes");
            }
            ResourceState state;
            state.output.resource_id = budget.resource_id;
            state.output.device = budget.device;
            state.output.usable_budget_bytes = budget.usable_budget_bytes;
            state.output.fixed_bytes = budget.fixed_bytes;
            state.output.transfer_staging_bytes =
                budget.transfer_staging_bytes;
            state.output.safety_reserve_bytes = budget.safety_reserve_bytes;
            state.output.live_copies_per_layer.assign(layer_count, 0);
            state.output.shadow_copies_per_layer.assign(layer_count, 0);
            if (resourceUsed(state.output) > budget.usable_budget_bytes)
            {
                std::ostringstream error;
                error
                    << "ExpertOverlay fixed/staging/reserve BOM already exceeds "
                       "physical resource '"
                    << budget.resource_id << "': used="
                    << resourceUsed(state.output) << " usable="
                    << budget.usable_budget_bytes << " fixed="
                    << state.output.fixed_bytes << " staging="
                    << state.output.transfer_staging_bytes << " reserve="
                    << state.output.safety_reserve_bytes;
                throw std::invalid_argument(error.str());
            }
            resources.push_back(std::move(state));
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
                    !participant_ids.insert(participant.participant_id).second)
                {
                    throw std::invalid_argument(
                        "ExpertOverlay capacity participants require unique ids, known resources, and positive shadow capacity");
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
                tier, footprints, resource_index, resources);
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
                    resources[resource_index.at(participant.resource_id)].output;
                for (std::size_t layer = 0; layer < layer_count; ++layer)
                {
                    if (participant.shadow_slots_per_layer >
                        static_cast<std::size_t>(
                            std::numeric_limits<int>::max() -
                            resource.shadow_copies_per_layer[layer]))
                    {
                        throw std::overflow_error(
                            "ExpertOverlay shadow copy count exceeds int");
                    }
                    resource.shadow_copies_per_layer[layer] +=
                        static_cast<int>(participant.shadow_slots_per_layer);
                }
            }
        }

        std::vector<int> unassigned(
            layer_count, input.num_experts);

        /*
         * Fixed quotas are exact constraints, so reserve and charge them before
         * any automatic tier consumes headroom. This prevents a more-preferred
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
                            footprints,
                            resource_index,
                            resources);
                    }
                }
                unassigned[layer] -= quota;
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

        result.physical_resources.reserve(resources.size());
        for (auto &resource : resources)
        {
            resource.output.used_bytes = resourceUsed(resource.output);
            if (resource.output.used_bytes >
                resource.output.usable_budget_bytes)
            {
                throw std::logic_error(
                    "ExpertOverlay capacity resolver committed an over-budget resource");
            }
            resource.output.remaining_bytes =
                resource.output.usable_budget_bytes -
                resource.output.used_bytes;
            result.physical_resources.push_back(std::move(resource.output));
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
