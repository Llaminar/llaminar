/**
 * @file MoEOverlayDeviceControllerTopology.h
 * @brief Frozen leader/follower topology for an all-GPU ExpertOverlay authority.
 *
 * ExpertOverlay has one logical authority in every multi-participant MoE cell.
 * When every routed-expert participant is a GPU, this type resolves that
 * authority into one continuation-root leader plus device-resident follower
 * groups. Homogeneous rank-local groups retain their native NCCL/RCCL control
 * collective; group roots exchange the small topology-wide policy record over
 * a separately certified node-local mapped fabric. No tier name, accelerator
 * vendor, MPI rank, or socket number has a privileged meaning.
 */

#pragma once

#include "MoEExpertOwnerMap.h"

#include <cstdint>
#include <span>
#include <string>
#include <vector>

namespace llaminar2
{
    /** Device transport used to mirror one leader decision inside a group. */
    enum class MoEOverlayDeviceControllerIntraGroupTransport : std::uint8_t
    {
        SingleParticipant = 0, ///< The group root is its only participant.
        NativeCollective,      ///< One homogeneous rank-local NCCL/RCCL group.
    };

    /** Device transport joining one follower-group root to the sole leader. */
    enum class MoEOverlayDeviceControllerInterGroupTransport : std::uint8_t
    {
        LeaderLocal = 0, ///< This is the group containing the authority leader.
        NodeLocalMapped, ///< Fixed device-owned records in mapped shared pages.
    };

    /**
     * @brief One native-control group beneath the topology-wide authority.
     *
     * A group is never a second planner. Only @ref leader_participant_id in the
     * enclosing topology computes policy. This root validates or relays the
     * leader's immutable command and every other member follows through the
     * declared intra-group transport.
     */
    struct MoEOverlayDeviceControllerGroup
    {
        int group_id = -1;             ///< Dense stable group ordinal.
        int tier_index = -1;           ///< Placement tier index, not preference.
        int tier_priority = 0;         ///< Opaque integer ordering from the plan.
        int domain_ordinal = -1;       ///< Stable index in `plan.domains`.
        std::string domain_name;        ///< Exact declarative domain identity.
        int root_participant_id = -1;  ///< Device follower root for this group.
        int root_world_rank = -1;      ///< MPI rank owning the group root.
        DeviceId root_device = DeviceId::invalid(); ///< Exact local root device.
        std::vector<int> participant_ids; ///< Sorted global participant ids.
        MoEOverlayDeviceControllerIntraGroupTransport intra_group_transport =
            MoEOverlayDeviceControllerIntraGroupTransport::SingleParticipant;
        MoEOverlayDeviceControllerInterGroupTransport inter_group_transport =
            MoEOverlayDeviceControllerInterGroupTransport::NodeLocalMapped;

        /** @return Whether immutable topology fields name a usable group. */
        [[nodiscard]] bool valid() const noexcept;

        /** @return Whether this group contains @p participant_id. */
        [[nodiscard]] bool contains(int participant_id) const noexcept;
    };

    /**
     * @brief Complete immutable device-controller topology for one model.
     *
     * The leader is resolved from the continuation domain and its logical root,
     * so moving CUDA/ROCm cards between ranks or reversing which vendor owns
     * continuation changes data only. `topology_fingerprint` authenticates the
     * exact participant/group/priority recipe embedded by retained graphs.
     */
    struct MoEOverlayDeviceControllerTopology
    {
        int leader_participant_id = -1; ///< Sole policy and epoch-CAS writer.
        int leader_group_id = -1;       ///< Group containing the leader.
        int leader_world_rank = -1;     ///< Rank owning the leader GPU.
        DeviceId leader_device = DeviceId::invalid(); ///< Exact leader GPU.
        std::vector<MoEExpertOwnerParticipant> participants; ///< Canonical owner-map order.
        std::vector<MoEOverlayDeviceControllerGroup> groups; ///< Canonical group order.
        std::uint64_t topology_fingerprint = 0u; ///< Non-zero frozen identity.

        /** @return Whether every topology invariant needed by graph wiring holds. */
        [[nodiscard]] bool valid() const noexcept;

        /** @return Group containing @p participant_id, or null when absent. */
        [[nodiscard]] const MoEOverlayDeviceControllerGroup *groupForParticipant(
            int participant_id) const noexcept;

        /** @return Whether a mapped inter-group device control fabric is required. */
        [[nodiscard]] bool usesNodeLocalMappedControl() const noexcept;
    };

    /** Inputs that bind declarative ranks to physical-node locality. */
    struct MoEOverlayDeviceControllerTopologyOptions
    {
        /**
         * Node id indexed by world rank, copied from `IMPITopology`.
         * Cross-rank locality is never inferred from rank numbers, sockets, or
         * the string `localhost`.
         */
        std::span<const int> world_rank_node_ids;
    };

    /**
     * @brief Resolve one node-local all-GPU leader/follower control topology.
     *
     * The function accepts only a frozen device-resident ExpertOverlay plan and
     * its exact immutable owner-map catalogue. Any CPU participant, missing
     * world-rank identity, cross-node edge, malformed continuation root, or
     * inconsistent participant catalogue is rejected. This is intentional:
     * an unsupported all-GPU fabric must fail setup rather than select host
     * policy as a fallback.
     *
     * @param plan Frozen placement, priority, domain, and continuation policy.
     * @param owner_map Stable global participant catalogue for the first epoch.
     * @param options Physical rank-to-node binding from MPI topology discovery.
     * @return Complete immutable device-controller topology.
     * @throws std::invalid_argument when ownership or topology is malformed.
     * @throws std::logic_error when the requested all-GPU control edge cannot
     *         be represented by the node-local production fabric.
     */
    [[nodiscard]] MoEOverlayDeviceControllerTopology
    resolveMoEOverlayDeviceControllerTopology(
        const MoERoutedExpertPlacementPlan &plan,
        const MoEExpertOwnerMap &owner_map,
        const MoEOverlayDeviceControllerTopologyOptions &options);

    /** @return Stable diagnostic spelling for @p transport. */
    [[nodiscard]] const char *toString(
        MoEOverlayDeviceControllerIntraGroupTransport transport) noexcept;

    /** @return Stable diagnostic spelling for @p transport. */
    [[nodiscard]] const char *toString(
        MoEOverlayDeviceControllerInterGroupTransport transport) noexcept;
} // namespace llaminar2
