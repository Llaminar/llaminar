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

#include "MoEOverlayCapacityResolver.h"

#include <cstddef>
#include <string>
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
    struct MoEOverlayBoundPhysicalMemoryBudget
    {
        int world_rank = -1;
        DeviceId device = DeviceId::invalid();
        std::string resource_id;
        std::size_t usable_budget_bytes = 0;
        std::size_t fixed_bytes = 0;
        /** Extra setup-owned transfer bytes outside the canonical fabric BOM. */
        std::size_t additional_transfer_staging_bytes = 0;
        std::size_t safety_reserve_bytes = 0;
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

    /** @brief Immutable materialization policy shared by admission and fabric setup. */
    struct MoEOverlayCapacityAdmissionPolicy
    {
        /** Canonical bounded transaction admits one incoming expert per endpoint/layer. */
        static constexpr std::size_t kProductionShadowSlots = 1;
        /** Canonical streaming chunk used by local and remote transfer lanes. */
        static constexpr std::size_t kProductionStagingBytes =
            4u * 1024u * 1024u;
        /** Free VRAM retained after graph, live experts, staging, and shadows. */
        static constexpr std::size_t kProductionGpuSafetyMarginBytes =
            128u * 1024u * 1024u;

        /** Whether runtime promotion/demotion resources will be materialized. */
        bool materialize_migration_fabric = false;
        /** Number of inactive RCU arrival slots owned by each endpoint/layer. */
        std::size_t shadow_slots_per_endpoint_layer = 1;
        /** Bounded bytes in each persistent conversion/transport staging chunk. */
        std::size_t staging_capacity_bytes = 0;
        /** Whether the fabric also materializes cross-rank MPI projection lanes. */
        bool distributed_transport = false;
        /** Exact overlay communicator size, including relay-only ranks. */
        int overlay_world_size = 1;
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
