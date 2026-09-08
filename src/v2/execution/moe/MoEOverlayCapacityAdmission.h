/**
 * @file MoEOverlayCapacityAdmission.h
 * @brief Bound-topology adapter for exact ExpertOverlay capacity admission.
 *
 * The byte resolver intentionally knows nothing about orchestration domains or
 * MPI ranks.  Production topology, however, names logical participants through
 * routed domains.  This file provides the single typed join between those two
 * authorities: every bound `(world rank, local device)` maps to one physical
 * memory resource, and every tier becomes a strict integer-priority capacity
 * request.  Tier labels remain opaque identities throughout this conversion.
 */

#pragma once

#include "DeviceMoERebalanceWorkspaceContract.h"
#include "DeviceMoETransferSlotDirectory.h"
#include "MoEOverlayCapacityResolver.h"
#include "planning/PhysicalMemoryAuthority.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace llaminar2
{
    /**
     * @brief Setup-time memory authority for one bound process-local device.
     *
     * `world_rank` and `device` are the typed topology join key. `resource_id`
     * is retained only as a stable diagnostic/distributed-plan identity; it is
     * never parsed to recover topology. Multiple logical tier participants may
     * reference the same entry and are then charged to its one shared budget.
     */
    class MoEOverlayBoundPhysicalMemoryBudget final
    {
    public:
        /** @brief Bind a certified BOM to its stable distributed identity. */
        MoEOverlayBoundPhysicalMemoryBudget(
            std::string resource_id,
            PhysicalMemoryAdmissionCertificate certificate)
            : resource_id_(std::move(resource_id))
        {
            if (resource_id_.empty() ||
                certificate.bom().resource().world_rank < 0)
            {
                throw std::invalid_argument(
                    "ExpertOverlay bound memory requires a rank-bound certified BOM and diagnostic identity");
            }
            identity_ = {
                .world_rank = certificate.bom().resource().world_rank,
                .device = certificate.bom().resource().device,
            };
            PhysicalMemoryPlanBuilder builder;
            builder.add(certificate.bom());
            authority_ = std::make_shared<
                const PhysicalMemoryPlanAdmissionCertificate>(
                builder.build());
        }

        /**
         * @brief Bind one resource view to a shared topology-wide authority.
         * @param resource_id Stable distributed join identity.
         * @param authority Complete admitted CPU/GPU topology.
         * @param identity Exact allocator exposed by this view.
         */
        MoEOverlayBoundPhysicalMemoryBudget(
            std::string resource_id,
            std::shared_ptr<const PhysicalMemoryPlanAdmissionCertificate>
                authority,
            PhysicalMemoryAllocatorIdentity identity)
            : resource_id_(std::move(resource_id)),
              authority_(std::move(authority)),
              identity_(identity)
        {
            if (resource_id_.empty() || !authority_ ||
                identity_.world_rank < 0 || !authority_->plan().find(identity_))
            {
                throw std::invalid_argument(
                    "ExpertOverlay bound memory view requires an admitted rank-bound physical resource");
            }
        }

        /** @return Rank owning the physical allocator. */
        [[nodiscard]] int worldRank() const noexcept
        {
            return identity_.world_rank;
        }

        /** @return CPU/CUDA/ROCm allocator identity. */
        [[nodiscard]] DeviceId device() const noexcept
        {
            return identity_.device;
        }

        /** @return Stable logical join identity; never parsed for topology. */
        [[nodiscard]] const std::string &resourceId() const noexcept
        {
            return resource_id_;
        }

        /** @return Immutable complete base admission proof. */
        [[nodiscard]] PhysicalMemoryAdmissionCertificate certificate() const
        {
            return authority_->certificateFor(identity_);
        }

        /** @return Canonical BOM owned by the shared topology authority. */
        [[nodiscard]] const PhysicalMemoryBOM &bom() const noexcept
        {
            return *authority_->plan().find(identity_);
        }

        /** @return Shared complete CPU/GPU admission authority. */
        [[nodiscard]] const std::shared_ptr<
            const PhysicalMemoryPlanAdmissionCertificate> &authority()
            const noexcept
        {
            return authority_;
        }

        /** @return Sampled allocatable bytes certified by this resource. */
        [[nodiscard]] std::size_t usableBudgetBytes() const noexcept
        {
            return bom().resource()
                .admission_available_bytes;
        }

        /** @return Incremental base bytes excluding all typed staging owners. */
        [[nodiscard]] std::size_t fixedBytes() const noexcept
        {
            const auto &memory = bom();
            return memory.incrementalBytes() -
                   memory.bytes(PhysicalMemoryOwner::WeightLoadStaging) -
                   memory.bytes(
                       PhysicalMemoryOwner::ActivationTransportStaging) -
                   memory.bytes(PhysicalMemoryOwner::ExpertMigrationStaging);
        }

        /** @return Setup staging already present in the bound base BOM. */
        [[nodiscard]] std::size_t stagingBytes() const noexcept
        {
            const auto &memory = bom();
            return memory.bytes(PhysicalMemoryOwner::WeightLoadStaging) +
                   memory.bytes(
                       PhysicalMemoryOwner::ActivationTransportStaging) +
                   memory.bytes(PhysicalMemoryOwner::ExpertMigrationStaging);
        }

        /**
         * @brief Rebind the same charges to a new allocator observation.
         * @param available_bytes Newly sampled allocatable bytes.
         * @return Newly certified immutable bound budget.
         */
        [[nodiscard]] MoEOverlayBoundPhysicalMemoryBudget
        withAvailableBytes(std::size_t available_bytes) const
        {
            const auto &source = bom();
            PhysicalMemoryBOMBuilder builder(
                PhysicalMemoryResource{
                    .world_rank = worldRank(),
                    .device = device(),
                    .total_bytes = source.resource().total_bytes,
                    .admission_available_bytes = available_bytes,
                },
                source);
            return MoEOverlayBoundPhysicalMemoryBudget(
                resource_id_,
                PhysicalMemoryAdmissionCertificate(builder.build()));
        }

    private:
        std::string resource_id_;
        std::shared_ptr<const PhysicalMemoryPlanAdmissionCertificate>
            authority_;
        PhysicalMemoryAllocatorIdentity identity_;
    };

    /** @brief One logical endpoint after domain participants are rank-bound. */
    struct MoEOverlayBoundTierParticipant
    {
        int participant_id = -1;
        int tier_index = -1;
        int domain_participant_index = -1;
        int world_rank = -1;
        DeviceId device = DeviceId::invalid();
    };

    /** @brief One exact staging charge against a physical rank/device authority. */
    struct MoEOverlayTransferStagingCharge
    {
        int world_rank = -1;
        DeviceId device = DeviceId::invalid();
        std::size_t bytes = 0;
    };

    /**
     * @brief Physical storage implementation selected for Dynamic movement.
     *
     * The single-rank homogeneous GPU controller retains transfer arrivals in
     * graph-owned directories. Arbitrary-tier and heterogeneous controllers
     * use RCU shadow banks plus the physical residency fabric. These are
     * mutually exclusive allocation lifecycles, not two booleans that may be
     * accidentally enabled together.
     */
    enum class MoEOverlayMigrationStorageKind : std::uint8_t
    {
        Disabled = 0,
        DeviceTransferDirectory,
        PhysicalResidencyFabric,
    };

    /** @return Stable diagnostic name for one migration storage lifecycle. */
    [[nodiscard]] inline constexpr const char *toString(
        MoEOverlayMigrationStorageKind kind) noexcept
    {
        switch (kind)
        {
        case MoEOverlayMigrationStorageKind::Disabled:
            return "disabled";
        case MoEOverlayMigrationStorageKind::DeviceTransferDirectory:
            return "device_transfer_directory";
        case MoEOverlayMigrationStorageKind::PhysicalResidencyFabric:
            return "physical_residency_fabric";
        }
        return "invalid";
    }

    /** @brief Immutable materialization policy shared by admission and fabric setup. */
    struct MoEOverlayCapacityAdmissionPolicy
    {
        /** One closed cycle can deliver at most one expert to an endpoint/layer. */
        static constexpr std::size_t
            kProductionShadowSlotsPerConcurrentCycle = 1;
        /** Canonical streaming chunk used by local and remote transfer lanes. */
        static constexpr std::size_t kProductionStagingBytes =
            4u * 1024u * 1024u;
        /** Exact and exclusive physical movement-storage implementation. */
        MoEOverlayMigrationStorageKind migration_storage =
            MoEOverlayMigrationStorageKind::Disabled;
        /** Exact graph-owned directory capacity; zero outside that lifecycle. */
        DeviceMoETransferSlotDirectory::BufferedCapacity
            device_transfer_directory_capacity;
        /**
         * Exact persistent maintenance-graph workspace identity.
         *
         * Present if and only if @ref migration_storage selects the homogeneous
         * device transfer directory. The codebook-dependent payload stride is
         * bound later from the authenticated layer manifest; every other size
         * discriminator is frozen here before expert quotas are resolved.
         */
        std::optional<DeviceMoERebalanceWorkspaceCapacity>
            device_rebalance_workspace_capacity;
        /** Logical per-layer arrival cap; zero when no migration fabric exists. */
        std::size_t shadow_slots_per_endpoint_layer = 0;
        /** Bounded bytes in each persistent conversion/transport staging chunk. */
        std::size_t staging_capacity_bytes = 0;
        /**
         * Maximum closed migration cycles admitted in one publication wave.
         *
         * Admission and physical-fabric construction use this same value to
         * price and materialize every independent staging/event/command lane.
         * All such lanes are enqueued asynchronously; their exact streams come
         * from the separately accounted @ref maximum_execution_streams pool.
         */
        std::size_t maximum_concurrent_cycles = 1;
        /**
         * Maximum setup-owned background GPU streams per participant copy.
         *
         * Logical cycles retain independent staging and completion events;
         * compatible cycles share this bounded stream pool. Admission and the
         * physical fabric carry the same value so automatic capacity, preflight,
         * and runtime materialization cannot disagree about queue geometry.
         */
        std::size_t maximum_execution_streams = 1;
        /**
         * Maximum cycles from one wave that may target the same model layer.
         *
         * Transport lanes are sized by @ref maximum_concurrent_cycles because
         * cycles from different layers still execute in parallel.  An
         * endpoint/layer shadow bank only receives at most this many arrivals,
         * so charging it for the global wave width would evict live experts
         * without providing usable concurrency.
         */
        std::size_t maximum_cycles_per_layer = 1;
        /** Whether the fabric also materializes cross-rank MPI projection lanes. */
        bool distributed_transport = false;
        /** Exact overlay communicator size, including relay-only ranks. */
        int overlay_world_size = 1;

        /** @return Whether runtime movement owns any physical storage. */
        [[nodiscard]] constexpr bool migrationEnabled() const noexcept
        {
            return migration_storage !=
                   MoEOverlayMigrationStorageKind::Disabled;
        }

        /** @return Whether graph-owned homogeneous GPU directories are selected. */
        [[nodiscard]] constexpr bool usesDeviceTransferDirectory() const noexcept
        {
            return migration_storage ==
                   MoEOverlayMigrationStorageKind::DeviceTransferDirectory;
        }

        /** @return Whether the arbitrary-tier RCU physical fabric is selected. */
        [[nodiscard]] constexpr bool usesPhysicalResidencyFabric() const noexcept
        {
            return migration_storage ==
                   MoEOverlayMigrationStorageKind::PhysicalResidencyFabric;
        }

        /**
         * @brief Return the worst-case inactive-slot demand of the cycle BOM.
         *
         * A valid closed migration cycle visits a participant at most once.
         * Cycles in different layers use different endpoint/layer banks, so
         * their global transport concurrency does not add to one bank's slot
         * demand.  Setup therefore charges the smaller of the global wave cap
         * and the per-layer scheduling cap. Keeping this equation beside the
         * lane budget prevents scheduling, preflight, and materialization from
         * drifting apart when either public tuning changes.
         */
        [[nodiscard]] static constexpr std::size_t
        requiredShadowSlotsPerEndpointLayer(
            std::size_t concurrent_cycles,
            std::size_t cycles_per_layer) noexcept
        {
            return std::min(concurrent_cycles, cycles_per_layer) *
                kProductionShadowSlotsPerConcurrentCycle;
        }
    };

    /**
     * @brief Convert a hardware-bound placement plan into exact resolver input.
     *
     * This adapter is deterministic and device-free. It rejects unbound
     * participants, tensor-sharded whole-expert tiers, missing physical
     * budgets, and ambiguous duplicate resource bindings. Automatic/fixed
     * quota mode comes from the presence of setup-resolved per-layer quotas;
     * participant copy multiplicity comes only from the domain's typed compute
     * policy. Neither a tier's label nor declaration position affects order.
     */
    class MoEOverlayCapacityAdmission final
    {
    public:
        /**
         * @brief Select the exclusive migration-storage lifecycle from topology.
         *
         * Dynamic movement inside one rank-local homogeneous GPU tier uses the
         * captured device transfer directory. Every heterogeneous, multi-tier,
         * cross-rank, or host-coordinated topology uses the physical residency
         * fabric. Non-Dynamic modes own no movement storage.
         *
         * @param plan Normalized hardware-bound ExpertOverlay plan.
         * @param mode Requested durable residency-maintenance mode.
         * @return Exact physical storage implementation.
         * @throws std::invalid_argument when Dynamic sees an unresolved or
         *         malformed authority topology.
         */
        [[nodiscard]] static MoEOverlayMigrationStorageKind
        selectMigrationStorage(
            const MoERoutedExpertPlacementPlan &plan,
            MoERebalanceRuntimeMode mode);

        /**
         * @brief Enumerate logical endpoints with stable owner-map participant ids.
         * @param plan Hardware-bound tier/domain topology.
         * @return Tier-major, domain-participant-major endpoint descriptors.
         * @throws std::invalid_argument for unbound or missing domains.
         */
        [[nodiscard]] static std::vector<MoEOverlayBoundTierParticipant>
        boundParticipants(const MoERoutedExpertPlacementPlan &plan);

        /**
         * @brief Price every persistent staging allocation in the real fabric.
         *
         * Device chunks are charged to their GPU. Pinned chunks and bounded MPI
         * payload lanes are charged to the rank's CPU memory authority. Peer
         * lanes allocate no bulk staging and therefore contribute zero bytes.
         *
         * @param plan Hardware-bound tier/domain topology.
         * @param world_size Exact overlay communicator size.
         * @param policy Migration materialization and chunk policy.
         * @return Sorted nonzero rank/device staging charges.
         * @throws std::invalid_argument for inconsistent policy/topology.
         * @throws std::overflow_error when the materialized BOM overflows.
         */
        [[nodiscard]] static std::vector<MoEOverlayTransferStagingCharge>
        transferStagingBOM(
            const MoERoutedExpertPlacementPlan &plan,
            int world_size,
            const MoEOverlayCapacityAdmissionPolicy &policy);

        /**
         * @brief Build the complete pure input consumed by the byte resolver.
         *
         * @param plan Hardware-bound tier/domain topology. Placements may be
         *        empty because admission normally precedes cold-start planning.
         * @param num_experts Exact routed-expert count in every manifest layer.
         * @param layer_weight_manifest Authenticated GGUF projection contract.
         * @param physical_budgets Complete bound physical memory authorities.
         * @param policy Shadow/materialization policy shared with the fabric.
         * @return Fully joined, deterministic capacity-resolver input.
         * @throws std::invalid_argument for incomplete or ambiguous topology.
         * @throws std::overflow_error when a compatibility byte cap overflows.
         */
        [[nodiscard]] static MoEOverlayCapacityResolverInput buildResolverInput(
            const MoERoutedExpertPlacementPlan &plan,
            int num_experts,
            const std::vector<MoEOverlayLayerWeightManifest> &layer_weight_manifest,
            const std::vector<MoEOverlayBoundPhysicalMemoryBudget> &physical_budgets,
            const MoEOverlayCapacityAdmissionPolicy &policy = {});

        /**
         * @brief Resolve capacity and install its exact quotas into a plan copy.
         *
         * @param plan Hardware-bound tier/domain topology.
         * @param num_experts Exact routed-expert count per layer.
         * @param layer_weight_manifest Authenticated GGUF projection contract.
         * @param physical_budgets Complete bound physical memory authorities.
         * @param policy Shadow/materialization policy shared with the fabric.
         * @return Placement plan carrying authoritative per-layer tier quotas.
         * @throws std::invalid_argument or std::overflow_error when admission
         *         cannot produce a complete, physically feasible plan.
         */
        [[nodiscard]] static MoERoutedExpertPlacementPlan resolveAndInstall(
            const MoERoutedExpertPlacementPlan &plan,
            int num_experts,
            const std::vector<MoEOverlayLayerWeightManifest> &layer_weight_manifest,
            const std::vector<MoEOverlayBoundPhysicalMemoryBudget> &physical_budgets,
            const MoEOverlayCapacityAdmissionPolicy &policy = {});
    };
} // namespace llaminar2
