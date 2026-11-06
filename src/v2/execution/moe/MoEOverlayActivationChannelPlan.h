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
#include "MoEOverlayInferenceOriginalSlot.h"
#include "MoEOverlayNodeLocalActivationLayout.h"
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
        /** Combined exact endpoint grants when both ends belong to one rank. */
        std::vector<MoEOverlayActivationLocalLaneBinding> rank_local_lanes;
        /** Exact shared mapping consumed unchanged by both endpoint ranks. */
        MoEOverlayNodeLocalActivationLayout mapping_layout;

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
        /** Endpoint-private epoch-grant bytes retained by one logical lane. */
        std::size_t device_grant_bytes_per_lane = 0;
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
        /** Same cold original protocol roster; absent for host-ordered transports. */
        std::optional<MoEOverlayInferenceOriginalSlotGeometry> original_native_slots;
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
         * @brief Resolve native slot eligibility from the original bound placement.
         * @param placement Original device/host authority and participant declarations.
         * @param inventory Complete original world topology.
         * @param family Original retained graph family, including its draft ceiling.
         * @return Native geometry only for a world-one device-owned mapped follower group.
         * @throws std::invalid_argument For an incomplete family or bound topology.
         * This is a cold BOM declaration; it grants no execution or source authority.
         */
        [[nodiscard]] static std::optional<MoEOverlayInferenceOriginalSlotGeometry>
        originalNativeSlotGeometry(const MoERoutedExpertPlacementPlan& placement,
            const ClusterInventory& inventory,
            const MoEOverlayInferenceGraphFamilyIdentity& family);

        /**
         * @brief Count the actual main and learned follower cache row manifests.
         * @param maximum_rows Original admitted main activation-row ceiling.
         * @param family Original decode/request bounds and actual learned source-layer list.
         * @param prefill_rows Exact schedule-owned ordered retained prefill identities.
         * @return Cache count per original mapped follower endpoint, independent of K.
         * @throws std::invalid_argument For incomplete row/family declarations.
         * @throws std::overflow_error If the immutable cache roster overflows.
         */
        [[nodiscard]] static std::size_t followerCaptureCacheCount(int maximum_rows,
            const MoEOverlayInferenceGraphFamilyIdentity& family, std::span<const int> prefill_rows);

        /**
         * @brief Report whether any participant requires a distinct retained endpoint channel.
         * @param placement Frozen, hardware-bound routed placement.
         * @return True when capacity planning needs the gathered rank topology.
         * @throws std::invalid_argument when the continuation participant is
         *         absent or its rank identity is incomplete.
         *
         * This is the shared admission predicate for capacity and transport
         * preflight. Continuation shards and same-rank CPU work retain their original local
         * graph storage. Foreign GPU domains require their own endpoint lanes
         * even when their true world rank equals the continuation rank.
         */
        [[nodiscard]] static bool requiresActivationChannels(
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
