/**
 * @file MoEOverlayActivationChannelPlan.h
 * @brief Pure topology and memory plan for node-local ExpertOverlay activations.
 *
 * Capacity admission and transport preflight must describe the same physical
 * lanes.  This device-free planner is their shared authority: it groups remote
 * routed participants into rank-pair channels, records every endpoint-local
 * payload lane, and prices the persistent GPU staging matrices embedded by the
 * retained main/MTP graph families.  Inter-node participants are deliberately
 * absent because their MPI transport owns a different memory contract.
 */

#pragma once

#include "MoEOverlayCapacityAdmission.h"
#include "MoERoutedExpertPlacementPlan.h"
#include "execution/mpi_orchestration/DeviceInventory.h"

#include <cstddef>
#include <vector>

namespace llaminar2
{
    /** @brief One captured payload lane allocated on an exact endpoint device. */
    struct MoEOverlayActivationLocalLaneBinding
    {
        int participant_id = -1;
        DeviceId device = DeviceId::invalid();

        /** @return Whether the lane has a complete participant/device identity. */
        [[nodiscard]] bool valid() const noexcept
        {
            return participant_id >= 0 && device.is_valid() &&
                   (device.is_cpu() || device.is_gpu());
        }
    };

    /** @brief One node-local source/target-rank activation channel. */
    struct MoEOverlayNodeLocalActivationChannelPlan
    {
        int tier_index = -1;
        int domain_ordinal = -1;
        int source_world_rank = -1;
        int target_world_rank = -1;
        DeviceId source_device = DeviceId::invalid();
        std::vector<MoEOverlayActivationLocalLaneBinding> source_lanes;
        std::vector<MoEOverlayActivationLocalLaneBinding> target_lanes;

        /**
         * @brief Return target participant IDs in stable packet order.
         * @return Strictly increasing participant IDs shared by both ranks.
         */
        [[nodiscard]] std::vector<int> targetParticipantIds() const;

        /**
         * @brief Select the exact captured lanes owned by one endpoint rank.
         * @param world_rank Source or target MPI rank for this channel.
         * @return Source-side or target-side lane bindings.
         * @throws std::invalid_argument if the rank is not an endpoint.
         */
        [[nodiscard]] const std::vector<
            MoEOverlayActivationLocalLaneBinding> &
        localLanes(int world_rank) const;
    };

    /** @brief Immutable node-local channel topology and exact staging BOM. */
    struct MoEOverlayActivationChannelPlan
    {
        std::size_t row_capacity = 0;
        int d_model = 0;
        int top_k = 0;
        std::size_t graph_family_count = 0;
        std::size_t payload_matrix_bytes = 0;
        /** One max-shape FP32 original-route bank per mapped GPU follower. */
        std::size_t canonical_route_matrix_bytes = 0;
        std::vector<MoEOverlayNodeLocalActivationChannelPlan> channels;
        std::vector<MoEOverlayTransferStagingCharge> staging_charges;

        /**
         * @brief Sum the charge for one physical endpoint.
         * @param world_rank Owning MPI rank.
         * @param device Rank-local CPU/GPU identity.
         * @return Persistent activation payload bytes, or zero when unused.
         */
        [[nodiscard]] std::size_t stagingBytesFor(
            int world_rank,
            DeviceId device) const noexcept;
    };

    /** @brief Complete input for deterministic activation-channel planning. */
    struct MoEOverlayActivationChannelPlannerInput
    {
        const MoERoutedExpertPlacementPlan *placement_plan = nullptr;
        const ClusterInventory *cluster_inventory = nullptr;
        std::size_t row_capacity = 0;
        int d_model = 0;
        int top_k = 0;
        std::size_t graph_family_count = 0;
    };

    /**
     * @brief Build the shared node-local activation topology and memory bill.
     *
     * The logical continuation root is the sole producer. Every non-root
     * participant on another rank of the same physical node receives one
     * target lane, including peers in the continuation tier; the source owns a
     * matching lane on its root device. Each lane owns one reusable FP32 matrix
     * per retained graph family. Tier priority controls residency and is not a
     * transport-locality predicate.
     */
    class MoEOverlayActivationChannelPlanner final
    {
    public:
        /**
         * @brief Report whether any non-root participant lives on another rank.
         * @param placement Frozen, hardware-bound routed placement.
         * @return True when capacity planning needs the gathered rank topology.
         * @throws std::invalid_argument when the continuation participant is
         *         absent or its rank identity is incomplete.
         *
         * This is the shared admission predicate for capacity and transport
         * preflight. A participant's tier never changes whether activations
         * must cross a rank boundary.
         */
        [[nodiscard]] static bool hasRemoteRankParticipants(
            const MoERoutedExpertPlacementPlan &placement);

        /**
         * @brief Resolve all node-local channels and aggregate staging charges.
         * @param input Bound placement, cluster topology, and graph geometry.
         * @return Deterministic channels and rank/device-qualified memory BOM.
         * @throws std::invalid_argument for incomplete or inconsistent topology.
         * @throws std::overflow_error when byte arithmetic exceeds size_t.
         */
        [[nodiscard]] static MoEOverlayActivationChannelPlan plan(
            const MoEOverlayActivationChannelPlannerInput &input);
    };
} // namespace llaminar2
