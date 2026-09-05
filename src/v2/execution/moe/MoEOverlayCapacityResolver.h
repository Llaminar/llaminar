/**
 * @file MoEOverlayCapacityResolver.h
 * @brief Exact setup-time memory and live-quota resolver for ExpertOverlay.
 *
 * ExpertOverlay residency is meaningful only when every live expert, inactive
 * migration slot, transfer lane, and graph/KV allocation is
 * admitted against the physical participant that owns those bytes.  This file
 * defines the device-free resolver contract used to turn those inputs into one
 * immutable integer-priority quota plan and a complete per-resource bill of
 * materials (BOM).
 *
 * The resolver is deliberately independent of model loading and allocation.
 * Production setup supplies the authenticated GGUF layer manifest, inventory
 * limits, and the already-resolved fixed graph/KV bytes.  Tests can therefore
 * exhaust every codebook, topology, and failure mode without occupying a GPU.
 */

#pragma once

#include "MoEOverlayPhysicalResidencyFabric.h"
#include "MoERoutedExpertPlacementPlan.h"
#include "backends/DeviceId.h"
#include "planning/PhysicalMemoryAuthority.h"

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace llaminar2
{
    /** @brief Whether a tier's live quota is setup-resolved or explicitly fixed. */
    enum class MoEOverlayLiveQuotaMode : std::uint8_t
    {
        Automatic = 0, ///< Fill exact remaining capacity in strict priority order.
        FixedPerLayer = 1, ///< Admit the supplied per-layer quotas exactly.
    };

    /** @brief Physical copy multiplicity for one logical tier resident. */
    enum class MoEOverlayTierCopyPolicy : std::uint8_t
    {
        Apportioned = 0, ///< One complete copy, balanced across participants.
        Replicated = 1,  ///< One complete copy on every tier participant.
    };

    /** @brief Setup-time live-residency invariant applied before priority fill. */
    enum class MoEOverlayInitialResidencyPolicy : std::uint8_t
    {
        /** Strict priority may leave a declared endpoint initially empty. */
        PriorityFillOnly = 0,
        /** Every migration participant/layer begins with one outbound source. */
        MigrationSourcePerParticipant = 1,
    };

    /**
     * @brief Memory authority shared by one or more logical tier participants.
     *
     * `resource_id` identifies physical storage, not a tier.  Two logical tiers
     * or participants naming the same resource are consequently charged to one
     * common budget. `usable_budget_bytes` is already the minimum of discovered
     * allocatable memory and any explicit user limit.
     */
    class MoEOverlayPhysicalMemoryBudget final
    {
    public:
        /** @brief Bind one logical join identity to a certified physical BOM. */
        MoEOverlayPhysicalMemoryBudget(
            std::string resource_id,
            PhysicalMemoryAdmissionCertificate certificate)
            : resource_id_(std::move(resource_id)),
              certificate_(std::move(certificate))
        {
            if (resource_id_.empty())
                throw std::invalid_argument(
                    "ExpertOverlay physical memory identity cannot be empty");
        }

        /** @return Stable logical identity used by tier participants. */
        [[nodiscard]] const std::string &resourceId() const noexcept
        {
            return resource_id_;
        }

        /** @return Physical CPU/CUDA/ROCm allocator. */
        [[nodiscard]] DeviceId device() const noexcept
        {
            return certificate_.bom().resource().device;
        }

        /** @return Certified fixed/staging allocation authority. */
        [[nodiscard]] const PhysicalMemoryAdmissionCertificate &certificate()
            const noexcept
        {
            return certificate_;
        }

        /** @return Sampled allocatable capacity for this resource. */
        [[nodiscard]] std::size_t usableBudgetBytes() const noexcept
        {
            return certificate_.bom().resource()
                .admission_available_bytes;
        }

        /** @return Base non-staging incremental allocation bytes. */
        [[nodiscard]] std::size_t fixedBytes() const noexcept
        {
            const auto &bom = certificate_.bom();
            return bom.incrementalBytes() - transferStagingBytes();
        }

        /** @return Complete typed staging charge in this base certificate. */
        [[nodiscard]] std::size_t transferStagingBytes() const noexcept
        {
            const auto &bom = certificate_.bom();
            return bom.bytes(PhysicalMemoryOwner::WeightLoadStaging) +
                   bom.bytes(
                       PhysicalMemoryOwner::ActivationTransportStaging) +
                   bom.bytes(PhysicalMemoryOwner::ExpertMigrationStaging);
        }

        /** @return Same owner charges certified against a new availability. */
        [[nodiscard]] MoEOverlayPhysicalMemoryBudget withAvailableBytes(
            std::size_t available_bytes) const
        {
            const auto &source = certificate_.bom();
            PhysicalMemoryBOMBuilder builder(
                PhysicalMemoryResource{
                    .world_rank = source.resource().world_rank,
                    .device = source.resource().device,
                    .total_bytes = source.resource().total_bytes,
                    .admission_available_bytes = available_bytes,
                },
                source);
            return MoEOverlayPhysicalMemoryBudget(
                resource_id_,
                PhysicalMemoryAdmissionCertificate(builder.build()));
        }

    private:
        std::string resource_id_;
        PhysicalMemoryAdmissionCertificate certificate_;
    };

    /** @brief One logical participant's binding to a physical memory authority. */
    struct MoEOverlayTierCapacityParticipant
    {
        int participant_id = -1;
        std::string resource_id;
        /** Per-layer arrival bound; zero disables the migration arena. */
        std::size_t shadow_slots_per_layer = 1;
        /** Global concurrently live arrivals across all exact geometries. */
        std::size_t maximum_concurrent_shadow_slots = 1;
    };

    /**
     * @brief Capacity policy for one strict-priority residency tier.
     *
     * Fixed quotas contain exactly one entry per model layer. Automatic quotas
     * leave that vector empty and are filled in priority order. An optional
     * maximum vector is an upper bound in either mode; it never substitutes for
     * physical byte admission. The fallback tier is the final coverage owner
     * and must have the numerically largest unique priority. Tier labels are
     * opaque identities and never participate in admission or ordering.
     */
    struct MoEOverlayTierCapacityRequest
    {
        int tier_index = -1;
        std::string tier_name;
        int priority = 0;
        bool fallback = false;
        MoEOverlayLiveQuotaMode quota_mode =
            MoEOverlayLiveQuotaMode::Automatic;
        MoEOverlayTierCopyPolicy copy_policy =
            MoEOverlayTierCopyPolicy::Apportioned;
        std::vector<int> fixed_live_experts_per_layer;
        std::vector<int> max_live_experts_per_layer;
        std::vector<MoEOverlayTierCapacityParticipant> participants;
    };

    /** @brief Complete pure input required to resolve one model's live quotas. */
    struct MoEOverlayCapacityResolverInput
    {
        int num_experts = 0;
        /**
         * Migration calibration uses real reversible transfers and therefore
         * needs one source expert on every declared endpoint/layer. The seed
         * is initial residency only, not a permanent quota: the runtime
         * authority may drain that endpoint after certification.
         */
        MoEOverlayInitialResidencyPolicy initial_residency_policy =
            MoEOverlayInitialResidencyPolicy::PriorityFillOnly;
        std::vector<MoEOverlayLayerWeightManifest> layer_weight_manifest;
        std::vector<MoEOverlayPhysicalMemoryBudget> physical_budgets;
        std::vector<MoEOverlayTierCapacityRequest> tiers;
    };

    /**
     * @brief Exact allocated bytes for one complete gate/up/down expert.
     *
     * CPU bytes are the final NativeVNNI interleaved representation. GPU live
     * bytes are the migration-stable separated representation that can remain
     * resident after a CPU round trip. GPU shadow bytes match the deliberately
     * all-format-capable inactive slot allocated by the physical fabric.
     */
    struct MoEOverlayPreparedExpertFootprint
    {
        int layer_idx = -1;
        std::size_t cpu_live_bytes = 0;
        std::size_t cpu_shadow_bytes = 0;
        std::size_t gpu_live_bytes = 0;
        std::size_t gpu_shadow_bytes = 0;

        /** @return Live prepared bytes for a CPU, CUDA, or ROCm endpoint. */
        [[nodiscard]] std::size_t liveBytes(DeviceId device) const;

        /** @return Inactive-slot bytes for a CPU, CUDA, or ROCm endpoint. */
        [[nodiscard]] std::size_t shadowBytes(DeviceId device) const;
    };

    /** @brief Resolved live quota and participant copy counts for one tier. */
    struct MoEOverlayResolvedTierCapacity
    {
        int tier_index = -1;
        std::string tier_name;
        int priority = 0;
        bool fallback = false;
        bool has_live_residency = false; ///< At least one resolved physical live copy.
        std::vector<int> live_experts_per_layer;
        /** Participant-major `[participant][layer]` physical live-copy counts. */
        std::vector<std::vector<int>> participant_live_copies;
    };

    /** @brief Auditable memory BOM for one unique physical resource. */
    class MoEOverlayResolvedPhysicalMemory final
    {
    public:
        /** @brief Publish the final certified BOM and its copy geometry. */
        MoEOverlayResolvedPhysicalMemory(
            std::string resource_id,
            std::shared_ptr<const PhysicalMemoryPlanAdmissionCertificate>
                authority,
            PhysicalMemoryAllocatorIdentity identity,
            std::vector<int> live_copies_per_layer,
            std::vector<int> shadow_arrival_capacity_per_layer)
            : resource_id_(std::move(resource_id)),
              authority_(std::move(authority)),
              identity_(identity),
              live_copies_per_layer(std::move(live_copies_per_layer)),
              shadow_arrival_capacity_per_layer(
                  std::move(shadow_arrival_capacity_per_layer))
        {
            if (resource_id_.empty() || !authority_ ||
                !authority_->plan().find(identity_))
            {
                throw std::invalid_argument(
                    "Resolved ExpertOverlay memory requires a non-empty identity and shared admitted resource");
            }
        }

        /** @return Logical join identity used by participant plans. */
        [[nodiscard]] const std::string &resourceId() const noexcept
        {
            return resource_id_;
        }

        /** @return CPU/CUDA/ROCm allocator owning this plan. */
        [[nodiscard]] DeviceId device() const noexcept
        {
            return identity_.device;
        }

        /** @return Canonical resource BOM owned by the aggregate authority. */
        [[nodiscard]] const PhysicalMemoryBOM &bom() const noexcept
        {
            return *authority_->plan().find(identity_);
        }

        /** @return Sole topology-wide admission authority behind this view. */
        [[nodiscard]] const std::shared_ptr<
            const PhysicalMemoryPlanAdmissionCertificate> &authority()
            const noexcept
        {
            return authority_;
        }

        /** @return Complete bytes across all physical owner categories. */
        [[nodiscard]] std::size_t usedBytes() const noexcept
        {
            return bom().incrementalBytes();
        }

        /** @return Unassigned bytes after all fixed and expert owners. */
        [[nodiscard]] std::size_t remainingBytes() const noexcept
        {
            return bom().remainingBytes();
        }

        /** @return Sampled allocatable bytes for the final resource. */
        [[nodiscard]] std::size_t usableBudgetBytes() const noexcept
        {
            return bom().resource()
                .admission_available_bytes;
        }

        /** @return Final inactive expert-bank bytes. */
        [[nodiscard]] std::size_t shadowBytes() const
        {
            return bom().bytes(
                PhysicalMemoryOwner::ExpertShadowSlots);
        }

        /** @return Final live prepared routed-expert bytes. */
        [[nodiscard]] std::size_t liveExpertBytes() const
        {
            return bom().bytes(
                PhysicalMemoryOwner::RoutedExpertWeights);
        }

        /** @return Weight-load, activation, and migration staging bytes. */
        [[nodiscard]] std::size_t transferStagingBytes() const
        {
            const auto &memory = bom();
            return memory.bytes(PhysicalMemoryOwner::WeightLoadStaging) +
                   memory.bytes(
                       PhysicalMemoryOwner::ActivationTransportStaging) +
                   memory.bytes(PhysicalMemoryOwner::ExpertMigrationStaging);
        }

        /** @return Incremental non-expert, non-staging base bytes. */
        [[nodiscard]] std::size_t fixedBytes() const
        {
            return usedBytes() - transferStagingBytes() - shadowBytes() -
                   liveExpertBytes();
        }

        std::vector<int> live_copies_per_layer;
        /** Logical arrival bound; physical shared-arena bytes live in the BOM. */
        std::vector<int> shadow_arrival_capacity_per_layer;

    private:
        std::string resource_id_;
        std::shared_ptr<const PhysicalMemoryPlanAdmissionCertificate>
            authority_;
        PhysicalMemoryAllocatorIdentity identity_;
    };

    /** @brief Immutable capacity result consumed by placement and admission. */
    struct MoEOverlayResolvedCapacityPlan
    {
        int num_experts = 0;
        std::vector<MoEOverlayPreparedExpertFootprint> layer_footprints;
        std::vector<MoEOverlayResolvedTierCapacity> tiers;
        /** One immutable admission proof for every CPU/GPU allocator. */
        std::shared_ptr<const PhysicalMemoryPlanAdmissionCertificate>
            physical_memory_admission;
        std::vector<MoEOverlayResolvedPhysicalMemory> physical_resources;

        /** @return Resolved tier entry for an external stable tier index. */
        [[nodiscard]] const MoEOverlayResolvedTierCapacity *tier(
            int tier_index) const noexcept;

        /** @return Resolved BOM entry for one physical resource identity. */
        [[nodiscard]] const MoEOverlayResolvedPhysicalMemory *resource(
            const std::string &resource_id) const noexcept;
    };

    /**
     * @brief Resolve exact prepared-weight footprints and priority quotas.
     *
     * This is a setup-time, deterministic operation. It throws rather than
     * shrinking fixed quotas, overcommitting a fallback tier, accepting equal
     * tier priorities, or treating absent capacity as unbounded.
     */
    class MoEOverlayCapacityResolver final
    {
    public:
        /**
         * @brief Build one complete immutable capacity plan.
         * @param input Authenticated model manifest, physical BOM, and tiers.
         * @return Exact quotas, participant copy counts, and remaining bytes.
         * @throws std::invalid_argument for malformed or infeasible input.
         * @throws std::overflow_error when any byte calculation overflows.
         */
        [[nodiscard]] static MoEOverlayResolvedCapacityPlan resolve(
            const MoEOverlayCapacityResolverInput &input);

        /**
         * @brief Install resolved per-layer quotas into a placement-plan copy.
         *
         * Stable tier indices, opaque names, integer priorities, and coverage
         * roles must match exactly. Existing placements are retained so the
         * shared plan validator can prove that an explicit layout satisfies
         * the installed quotas; an unplaced plan can be passed directly to
         * @ref MoERoutedExpertPlacementPlanner afterwards.
         *
         * @param plan Bound declarative tier/domain plan.
         * @param capacity Exact physical capacity result for the same tiers.
         * @return A copied plan carrying authoritative layer quotas.
         * @throws std::invalid_argument for identity or geometry mismatch.
         */
        [[nodiscard]] static MoERoutedExpertPlacementPlan installResolvedQuotas(
            const MoERoutedExpertPlacementPlan &plan,
            const MoEOverlayResolvedCapacityPlan &capacity);

        /**
         * @brief Compute CPU/GPU allocated bytes for every manifest layer.
         * @param manifest Complete authenticated gate/up/down layer contracts.
         * @return Layer-ordered exact prepared allocation footprints.
         * @throws std::invalid_argument for an incomplete/non-contiguous layer.
         * @throws std::overflow_error when geometry exceeds addressable bytes.
         */
        [[nodiscard]] static std::vector<MoEOverlayPreparedExpertFootprint>
        preparedFootprints(
            const std::vector<MoEOverlayLayerWeightManifest> &manifest);
    };
} // namespace llaminar2
