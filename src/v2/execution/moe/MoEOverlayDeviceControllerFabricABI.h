/**
 * @file MoEOverlayDeviceControllerFabricABI.h
 * @brief Fixed-width mapped-page layout for the node-local GPU controller.
 *
 * This ABI is intentionally backend-neutral. CUDA and ROCm kernels see the
 * same physical POSIX pages through process-local device aliases. The leader
 * owns immutable topology metadata and command storage; every controller group
 * owns one disjoint record and one member-major collected-state region.
 */

#pragma once

#include "MoEOverlayDeviceControllerABI.h"
#include "DeviceMoERuntimeABI.h"
#include "DeviceMoEOverlayServiceTelemetry.h"

#include <cstddef>
#include <cstdint>
#include <type_traits>

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

    /** Binary identity of the node-local controller fabric (`MOCF`). */
    inline constexpr std::uint32_t kMoEOverlayDeviceControllerFabricMagic =
        0x46434f4du;

    /** Version of every fixed-width fabric record in this header. */
    inline constexpr std::uint32_t kMoEOverlayDeviceControllerFabricVersion =
        22u;

    /** Snapshot and durable history planes: decode, prefill, grouped verifier. */
    inline constexpr std::uint32_t
        kMoEOverlayDeviceControllerDemandPhaseCount = 3u;

    /** Decode, real prefill, and grouped-verifier service-cost planes. */
    inline constexpr std::uint32_t
        kMoEOverlayDeviceControllerEconomyServicePhaseCount =
            kMoEOverlayDeviceControllerDemandPhaseCount;

    /** Service-profile plane matching `ExpertHistogramSource::DecodeToken`. */
    inline constexpr std::uint32_t
        kMoEOverlayDeviceControllerEconomyDecodePhase = 0u;

    /** Service-profile plane matching `ExpertHistogramSource::PrefillChunk`. */
    inline constexpr std::uint32_t
        kMoEOverlayDeviceControllerEconomyPrefillPhase = 1u;

    /** Service-profile plane matching grouped MTP verifier execution. */
    inline constexpr std::uint32_t
        kMoEOverlayDeviceControllerEconomyGroupedVerifierPhase = 2u;

    /** Initial hysteresis word for an expert with no committed movement. */
    inline constexpr std::uint64_t
        kMoEOverlayDeviceControllerNeverMovedGeneration =
            UINT64_MAX;

    /** Legacy scalar Dynamic policy scratch extent; never a fabric storage bound. */
    inline constexpr std::uint32_t
        kMoEOverlayWholeExpertPolicyMaxParticipants =
            8u;

    /** Maximum routed layers addressable by fixed-width controller commands. */
    inline constexpr std::uint32_t
        kMoEOverlayDeviceControllerFabricMaxLayers = 256u;

    /** Maximum logical experts addressable in one mapped snapshot layer. */
    inline constexpr std::uint32_t
        kMoEOverlayDeviceControllerFabricMaxExperts = moe_runtime_abi::kMaxExperts;

    /** Setup-only host lifecycle; inference never mutates this state. */
    enum class MoEOverlayDeviceControllerFabricSetupState : std::uint32_t
    {
        Uninitialized = 0u,
        LayoutPublished = 1u,
        PagesPlaced = 2u,
        ControllerInitialized = 3u,
        Registered = 4u,
        Error = 5u,
    };

    /** Stable setup failure retained for every attaching local rank. */
    enum class MoEOverlayDeviceControllerFabricSetupError : std::uint32_t
    {
        None = 0u,
        InvalidLayout = 1u,
        MappingFailure = 2u,
        PlacementTimeout = 3u,
        RegistrationFailure = 4u,
        TopologyMismatch = 5u,
    };

    /** Host-evidence/device-policy handoff for one immutable economy profile. */
    enum class MoEOverlayDeviceControllerEconomyState : std::uint32_t
    {
        Empty = 0u, ///< Device policy must observe without moving.
        Ready = 1u, ///< Complete costs and policy are release-published.
        Error = 2u, ///< Terminal evidence publication failure.
    };

    /** One directed participant/layer movement price consumed by device policy. */
    struct alignas(16) MoEOverlayDeviceControllerMigrationCost
    {
        std::uint64_t transfer_and_repack_ns = 0u;
        std::uint64_t inference_interference_ns = 0u;
    };

    /**
     * @brief Immutable measured-economy identity and arithmetic policy.
     *
     * A maintenance thread writes every referenced array first, then performs
     * one system-release store of @ref state. The device leader acquires that
     * word before reading costs. No owner map, desired placement, or epoch is
     * host-publishable through this record.
     */
    struct alignas(64) MoEOverlayDeviceControllerEconomyHeader
    {
        std::uint32_t magic = kMoEOverlayDeviceControllerFabricMagic;
        std::uint32_t version = kMoEOverlayDeviceControllerFabricVersion;
        std::uint32_t state = static_cast<std::uint32_t>(
            MoEOverlayDeviceControllerEconomyState::Empty);
        std::uint32_t tier_count = 0u;
        std::uint32_t logical_target_count = 0u;
        std::uint32_t layer_count = 0u;
        std::uint32_t service_phase_count =
            kMoEOverlayDeviceControllerEconomyServicePhaseCount;
        /** Bit `N` means production service phase `N` is runtime-reachable. */
        std::uint32_t active_source_bits = 0u;
        std::uint64_t topology_fingerprint = 0u;
        std::uint64_t service_identity_fingerprint = 0u;
        std::uint64_t migration_identity_fingerprint = 0u;
        std::uint64_t publication_generation = 0u;
        std::uint32_t historical_window_weight = 0u;
        std::uint32_t current_window_weight = 0u;
        std::uint64_t minimum_residency_generations = 0u;
        std::uint64_t payoff_horizon_tokens = 0u;
        std::uint64_t minimum_net_benefit_ns = 0u;
        std::uint64_t reserved[3] = {};
    };

    /** Device-readable role bits for one immutable participant. */
    enum class MoEOverlayDeviceControllerParticipantFlags : std::uint32_t
    {
        None = 0u,
        AuthorityLeader = 1u << 0u,
        GroupRoot = 1u << 1u,
        /** Participant joins the continuation transaction's exact-epoch barrier. */
        InferenceEpochMember = 1u << 2u,
    };

    /**
     * @brief One setup rendezvous shared by all participating MPI ranks.
     *
     * The creator touches this bootstrap record. The performance-sensitive
     * controller and snapshot pages live in separately owned page ranges.
     * `*_ready` entries are indexed by `participating_world_ranks`, not by
     * world rank, so arbitrary sparse rank numbering remains representable.
     */
    struct alignas(64) MoEOverlayDeviceControllerFabricSetupHeader
    {
        std::uint32_t magic = kMoEOverlayDeviceControllerFabricMagic;
        std::uint32_t version = kMoEOverlayDeviceControllerFabricVersion;
        std::uint32_t state = static_cast<std::uint32_t>(
            MoEOverlayDeviceControllerFabricSetupState::Uninitialized);
        std::uint32_t error_code = static_cast<std::uint32_t>(
            MoEOverlayDeviceControllerFabricSetupError::None);
        std::uint64_t topology_fingerprint = 0u;
        std::uint64_t node_namespace = 0u;
        std::uint64_t mapping_bytes = 0u;
        std::uint32_t num_layers = 0u;
        std::uint32_t num_experts = 0u;
        /** Exact routed expert fan-out produced by one token in one layer. */
        std::uint32_t routed_experts_per_token = 0u;
        std::uint32_t participant_count = 0u;
        std::uint32_t group_count = 0u;
        std::uint32_t command_capacity = 0u;
        std::uint32_t participating_rank_count = 0u;
        std::int32_t leader_world_rank = -1;
        /** Root-relative immutable rank roster and exact publication lane offsets. */
        std::uint64_t participating_world_ranks_offset = 0u;
        std::uint64_t first_touch_ready_offset = 0u;
        std::uint64_t registration_ready_offset = 0u;
        /** Per-rank proof that calibration demand was discarded before Dynamic admission. */
        std::uint64_t demand_histogram_rebase_ready_offset = 0u;
        /** Setup region extent; leader-owned pages start after this complete region. */
        std::uint64_t storage_bytes = 0u;
        /** Global origin authenticated before another rank joins these pages. */
        std::int32_t first_model_layer = 0;
        std::uint32_t reserved0 = 0u;
        std::uint64_t reserved[3] = {};
    };

    /**
     * @brief Immutable offsets interpreted relative to one device alias.
     *
     * Offsets rather than host pointers make the layout identical in every MPI
     * process even when CUDA and HIP assign different virtual device aliases.
     */
    struct alignas(64) MoEOverlayDeviceControllerFabricLayoutHeader
    {
        std::uint32_t magic = kMoEOverlayDeviceControllerFabricMagic;
        std::uint32_t version = kMoEOverlayDeviceControllerFabricVersion;
        std::uint32_t participant_count = 0u;
        std::uint32_t group_count = 0u;
        std::uint32_t num_layers = 0u;
        std::uint32_t num_experts = 0u;
        std::uint32_t command_capacity = 0u;
        /** Dense count of configured routed tiers; labels remain irrelevant. */
        std::uint32_t tier_count = 0u;
        std::uint64_t topology_fingerprint = 0u;
        std::uint64_t mapping_bytes = 0u;
        std::uint64_t participant_metadata_offset = 0u;
        std::uint64_t group_layout_offset = 0u;
        std::uint64_t controller_header_offset = 0u;
        /** Dedicated cache lines for symmetric continuation epoch admission. */
        std::uint64_t inference_epoch_record_offset = 0u;
        /** Complete record and member/arrival tail, admitted before graph capture. */
        std::uint64_t inference_epoch_storage_bytes = 0u;
        std::uint64_t command_header_offset = 0u;
        std::uint64_t command_entries_offset = 0u;
        /** Immutable exact bytes in one complete packed expert, per layer. */
        std::uint64_t payload_bytes_per_layer_offset = 0u;
        /** Immutable loader-prepared `[layer][expert]` original typed targets. */
        std::uint64_t initial_route_targets_offset = 0u;
        /** Exact number of typed four-byte tags in the initial target table. */
        std::uint64_t initial_route_targets_words = 0u;
        /** Leader-owned `[phase][layer][expert]` accumulated routed demand. */
        std::uint64_t demand_history_offset = 0u;
        /** Exact number of 64-bit words in @ref demand_history_offset. */
        std::uint64_t demand_history_words = 0u;
        /** Host-measured, device-consumed immutable economy header. */
        std::uint64_t economy_header_offset = 0u;
        /** `[logical target][layer][phase]` existing scalar service price. */
        std::uint64_t economy_service_cost_offset = 0u;
        std::uint64_t economy_service_cost_words = 0u;
        /** `[source][destination][layer]` directed movement prices. */
        std::uint64_t economy_migration_cost_offset = 0u;
        std::uint64_t economy_migration_cost_entries = 0u;
        /** Device-owned `[layer][expert]` last committed movement generation. */
        std::uint64_t economy_last_moved_offset = 0u;
        std::uint64_t economy_last_moved_words = 0u;
        /** Minimum routed observations required before policy may move. */
        std::uint64_t minimum_window_activations = 1u;
        /** Maximum complete capacity-preserving cycles admitted per epoch. */
        std::uint32_t maximum_cycles_per_wave = 1u;
        /** Required maximum/minimum routed-load ratio for local skew work. */
        std::uint32_t dynamic_imbalance_threshold_per_mille = 0u;
        /** Required fractional objective reduction for every accepted cycle. */
        std::uint32_t dynamic_minimum_improvement_per_mille = 0u;
        /** Per-layer cycle cap; zero disables Dynamic movement. */
        std::uint32_t dynamic_maximum_cycles_per_layer = 0u;
        /** Per-wave command cap; zero means the mapped command capacity. */
        std::uint32_t dynamic_maximum_commands_per_wave = 0u;
        /** Digest binding the immutable per-layer payload geometry. */
        std::uint64_t payload_geometry_fingerprint = 0u;
        /** Converts layer activation history into a token payoff horizon. */
        std::uint32_t routed_experts_per_token = 0u;
        /** Model-global identity of compact controller row zero. */
        std::int32_t first_model_layer = 0;
        /** Exact policy vertex count; cohort members are not separate vertices. */
        std::uint32_t logical_target_count = 0u;
        /** Original continuation-root publisher of global selected-route demand. */
        std::uint32_t demand_publisher_participant_id = 0u;
        /** Original group containing that publisher; no root/member renumbering. */
        std::uint32_t demand_publisher_group_id = 0u;
        std::uint32_t reserved_target0 = 0u;
        /** Immutable original-target records, each reusing its group's member roster. */
        std::uint64_t logical_target_metadata_offset = 0u;
        std::uint64_t reserved[1] = {};
    };

    /**
     * @brief One logical policy vertex over an immutable original member extent.
     * Whole vertices name their original participant; a cohort names its one
     * typed tier tag and every member of its original native group. Roster
     * positions are private algorithm indices, never participant identities.
     */
    struct alignas(8) MoEOverlayDeviceControllerLogicalTargetMetadata
    {
        MoEOverlayRouteTarget target;
        std::uint32_t group_id = 0u;
        std::int32_t tier_index = -1;
        std::int32_t tier_priority = 0;
        std::uint32_t first_group_member = 0u;
        std::uint32_t member_count = 0u;
        std::uint64_t reserved = 0u;
        /** @param other Immutable original roster row. @return Exact target, member extent and policy identity. */
        constexpr bool operator==(const MoEOverlayDeviceControllerLogicalTargetMetadata &other) const noexcept
        {
            return target.word() == other.target.word() && group_id == other.group_id &&
                tier_index == other.tier_index && tier_priority == other.tier_priority &&
                first_group_member == other.first_group_member && member_count == other.member_count && reserved == other.reserved;
        }
    };
    static_assert(sizeof(MoEOverlayDeviceControllerLogicalTargetMetadata) == 32u);

    /** Immutable topology/policy metadata for one dense participant id. */
    struct alignas(64) MoEOverlayDeviceControllerParticipantMetadata
    {
        std::uint32_t participant_id = 0u;
        std::uint32_t group_id = 0u;
        std::int32_t tier_index = -1;
        std::int32_t tier_priority = 0;
        std::int32_t world_rank = -1;
        std::uint32_t device_type = 0u;
        std::int32_t device_ordinal = -1;
        std::uint32_t domain_participant_index = 0u;
        std::uint32_t flags = static_cast<std::uint32_t>(
            MoEOverlayDeviceControllerParticipantFlags::None);
        std::uint32_t reserved0 = 0u;
        std::uint64_t reserved[3] = {};
    };

    /**
     * @brief One participant-owned release lane for the complete epoch lifecycle.
     *
     * Each GPU writes only its own record. The group root acquires every member
     * at snapshot, inactive-bank preparation, selector publication, reader-
     * drained retirement, and transient restoration boundaries. This prevents
     * a group-level acknowledgement from masquerading as proof that all local
     * devices completed the corresponding operation.
     *
     * Snapshot evidence and `prepared_transaction` occupy the first cache line;
     * publication/retirement fields occupy the second. Inference writes only terminal readiness/progress receipts, outside its
     * arithmetic body; maintenance consumes the original mapped publication.
     */
    struct alignas(64) MoEOverlayDeviceControllerParticipantRecord
    {
        std::uint32_t magic = kMoEOverlayDeviceControllerFabricMagic;
        std::uint32_t version = kMoEOverlayDeviceControllerFabricVersion;
        std::uint32_t participant_id = 0u;
        std::uint32_t group_id = 0u;
        std::uint64_t topology_fingerprint = 0u;
        std::uint64_t snapshot_digest = 0u;
        std::uint64_t snapshot_observations = 0u;
        std::uint64_t snapshot_transaction = 0u;
        std::uint32_t status_code = static_cast<std::uint32_t>(
            MoEOverlayDeviceControllerError::None);
        /**
         * Last entered maintenance action, for failure diagnosis only. Concurrent
         * inference retirement-readiness probes must not overwrite this lane.
         * Zero means no authenticated maintenance action has entered.
         * Never use this observation to authorize a lifecycle transition.
         */
        std::uint32_t observed_action = 0u;
        /** Complete inactive runtime bank installed but not admission-visible. */
        std::uint64_t prepared_transaction = 0u;
        /** Local RCU selector and runtime metadata publish completion. */
        std::uint64_t published_transaction = 0u;
        /** Local old-bank readers drained; topology-wide reclaim may begin. */
        std::uint64_t retirement_ready_epoch = 0u;
        /** Old durable epoch reclaimed after this device's readers drained. */
        std::uint64_t retired_epoch = 0u;
        /** Current-batch LLEP transient state restored on this participant. */
        std::uint64_t restored_transaction = 0u;
        /** Controller transaction observed at action entry; diagnostic-only. */
        std::uint64_t observed_action_transaction = 0u;
        /** Original finite parent generation retaining this transaction's old bank. */
        std::uint64_t finite_parent_generation = 0u;
        /** Actual captured Arm/child-terminal/End progress, never a host heartbeat. */
        std::uint64_t finite_parent_progress = 0u;
        /** Release publication binds these two words to the original command. */
        std::uint64_t finite_deferred_transaction = 0u;
    };

    /** Host transport worker lifecycle for one device-controller group. */
    enum class MoEOverlayDeviceControllerTransportState : std::uint32_t
    {
        Idle = 0u,            ///< No immutable command has been acquired.
        CommandAcquired = 1u, ///< The exact device command passed validation.
        Prepared = 2u,        ///< Every local transfer/preparation edge is ready.
        Published = 3u,       ///< The local inactive execution bank is installed.
        Retired = 4u,         ///< The named prior durable epoch is reclaimed.
        Restored = 5u,        ///< Request-scoped LLEP arrivals are reclaimed.
        Error = 6u,           ///< The first physical transport failure is latched.
    };

    /**
     * @brief Group-local physical completion lane written only by transport.
     *
     * This record is deliberately distinct from
     * @ref MoEOverlayDeviceControllerGroupRecord. The group-root GPU remains
     * the sole writer of controller acknowledgements; a host worker may only
     * authenticate an immutable device-authored command and publish physical
     * readiness here. Device code acquires these words before advancing the
     * topology-wide epoch. No owner map, histogram, or policy result can be
     * represented in this ABI.
     */
    struct alignas(64) MoEOverlayDeviceControllerTransportRecord
    {
        std::uint32_t magic = kMoEOverlayDeviceControllerFabricMagic;
        std::uint32_t version = kMoEOverlayDeviceControllerFabricVersion;
        std::uint32_t group_id = 0u;
        std::uint32_t state = static_cast<std::uint32_t>(
            MoEOverlayDeviceControllerTransportState::Idle);
        std::uint64_t topology_fingerprint = 0u;
        /** Exact immutable command transaction accepted by the worker. */
        std::uint64_t command_transaction = 0u;
        /** Digest re-derived by the worker from the immutable entry bytes. */
        std::uint64_t command_digest = 0u;
        /** Transfer/preparation completion for @ref command_transaction. */
        std::uint64_t prepared_transaction = 0u;
        /** Inactive-bank installation completion for the same transaction. */
        std::uint64_t published_transaction = 0u;
        /** Prior durable epoch reclaimed after all exact readers drain. */
        std::uint64_t retired_epoch = 0u;
        /** LLEP transaction whose transient physical arrivals were reclaimed. */
        std::uint64_t restored_transaction = 0u;
        /** @ref MoEOverlayDeviceControllerError encoded by the worker. */
        std::uint32_t status_code = 0u;
        std::uint32_t reserved0 = 0u;
        std::uint64_t reserved[5] = {};
    };

    /**
     * @brief Immutable page range and membership of one controller group.
     *
     * Collected state is member-major. Each member publishes one live
     * residency plane; only the original demand publisher appends the two
     * remaining selected-route phase planes. Central member extent helpers own
     * this compact layout. Words use `moe_rebalance_policy::packCollectedState`
     * for selected demand, physical residency and durable authority flags.
     */
    struct alignas(64) MoEOverlayDeviceControllerFabricGroupLayout
    {
        std::uint32_t group_id = 0u;
        std::uint32_t participant_count = 0u;
        std::uint32_t root_participant_id = 0u;
        std::int32_t root_world_rank = -1;
        std::int32_t tier_index = -1;
        std::int32_t tier_priority = 0;
        std::uint32_t intra_group_transport = 0u;
        std::uint32_t inter_group_transport = 0u;
        /** Mapping-relative immutable member roster; count is participant_count. */
        std::uint64_t participant_ids_offset = 0u;
        std::uint64_t participant_records_offset = 0u;
        std::uint64_t participant_record_count = 0u;
        std::uint64_t group_record_offset = 0u;
        /** Host-transport-written, GPU-consumed physical readiness lane. */
        std::uint64_t transport_record_offset = 0u;
        std::uint64_t collected_state_offset = 0u;
        std::uint64_t collected_state_words = 0u;
        /** Participant-major mapped service snapshot publications. */
        std::uint64_t service_telemetry_offset = 0u;
        /** Cache-line-aligned bytes reserved for each participant record. */
        std::uint64_t service_telemetry_stride_bytes = 0u;
        /** Complete mapped bytes occupied by all participant publications. */
        std::uint64_t service_telemetry_bytes = 0u;
        std::uint64_t owned_page_begin = 0u;
        std::uint64_t owned_page_end = 0u;
        /** Complete original cohort over participant_ids, absent for whole owners. */
        MoEOverlayRouteTarget cohort_target;
        std::uint32_t reserved0 = 0u;
        std::uint64_t reserved[7] = {};
    };

#if defined(__CUDACC__) || defined(__HIPCC__)
#define LLAMINAR_MOE_FABRIC_EXTENT_HD __host__ __device__
#else
#define LLAMINAR_MOE_FABRIC_EXTENT_HD
#endif
    /**
     * @brief Resolve the exact immutable group roster from the admitted mapping.
     * @param base Exact host/device mapping alias. @param bytes Admitted mapping bytes.
     * @param group Immutable group extent. @return Null for malformed/out-of-range storage.
     */
    LLAMINAR_MOE_FABRIC_EXTENT_HD inline const std::uint32_t *moeOverlayGroupMemberIds(
        const void *base, std::uint64_t bytes,
        const MoEOverlayDeviceControllerFabricGroupLayout &group) noexcept
    {
        return group.participant_count ? moeOverlayExtentArray<std::uint32_t>(
            base, bytes, group.participant_ids_offset, group.participant_count) : nullptr;
    }
    /**
     * @brief Resolve original membership without interpreting a model-domain coordinate as a group index.
     * @param ids Same admitted original group roster. @param count Its exact live extent.
     * @param participant Canonical global participant identity. @param index Receives its group-local position.
     * @return True only for exactly one matching member; no domain coordinate is changed.
     */
    LLAMINAR_MOE_FABRIC_EXTENT_HD inline bool moeOverlayFindOriginalGroupMember(
        const std::uint32_t *ids, std::uint32_t count, std::uint32_t participant,
        std::uint32_t &index) noexcept
    {
        if (!ids || !count) return false;
        bool found = false;
        for (std::uint32_t member = 0u; member < count; ++member)
            if (ids[member] == participant)
            {
                if (found) return false;
                found = true;
                index = member;
            }
        return found;
    }

    /**
     * @brief Count only live member snapshot words.
     * @param layers Original compact layer extent. @param experts Original expert extent.
     * @param participant Original global member. @param publisher Original global demand producer.
     * @return Residency/authority plane, plus two phase planes only on the producer.
     */
    LLAMINAR_MOE_FABRIC_EXTENT_HD constexpr std::uint64_t moeOverlaySnapshotMemberWords(
        std::uint32_t layers, std::uint32_t experts, std::uint32_t participant,
        std::uint32_t publisher) noexcept
    {
        return std::uint64_t(layers) * experts *
            (participant == publisher ? kMoEOverlayDeviceControllerDemandPhaseCount : 1u);
    }

    /**
     * @brief Resolve one original member's live snapshot offset in its existing group.
     * @param layout Immutable admitted geometry. @param group Original group extent.
     * @param ids Same group's original immutable roster. @param member Original roster index.
     * @param offset Receives the exact word offset, never a capacity-derived stride.
     * @return False for foreign members or a layout inconsistent with the live word count.
     */
    LLAMINAR_MOE_FABRIC_EXTENT_HD inline bool moeOverlaySnapshotMemberOffset(
        const MoEOverlayDeviceControllerFabricLayoutHeader &layout,
        const MoEOverlayDeviceControllerFabricGroupLayout &group,
        const std::uint32_t *ids, std::uint32_t member, std::uint64_t &offset) noexcept
    {
        if (!ids || member >= group.participant_count || !layout.num_layers || !layout.num_experts)
            return false;
        std::uint64_t cursor = 0u;
        for (std::uint32_t index = 0u; index < group.participant_count; ++index)
        {
            if (ids[index] >= layout.participant_count) return false;
            if (index == member) offset = cursor;
            const auto words = moeOverlaySnapshotMemberWords(layout.num_layers, layout.num_experts,
                ids[index], layout.demand_publisher_participant_id);
            if (cursor > UINT64_MAX - words) return false;
            cursor += words;
        }
        return cursor == group.collected_state_words;
    }
#undef LLAMINAR_MOE_FABRIC_EXTENT_HD

    static_assert(std::is_trivially_copyable_v<
                  MoEOverlayDeviceControllerFabricSetupHeader>);
    static_assert(std::is_trivially_copyable_v<
                  MoEOverlayDeviceControllerEconomyHeader>);
    static_assert(std::is_trivially_copyable_v<
                  MoEOverlayDeviceControllerMigrationCost>);
    static_assert(std::is_trivially_copyable_v<
                  MoEOverlayDeviceControllerFabricLayoutHeader>);
    static_assert(std::is_trivially_copyable_v<
                  MoEOverlayDeviceControllerParticipantMetadata>);
    static_assert(std::is_trivially_copyable_v<
                  MoEOverlayDeviceControllerParticipantRecord>);
    static_assert(std::is_trivially_copyable_v<
                  MoEOverlayDeviceControllerTransportRecord>);
    static_assert(std::is_trivially_copyable_v<
                  MoEOverlayDeviceControllerFabricGroupLayout>);
    static_assert(
        sizeof(MoEOverlayDeviceControllerFabricSetupHeader) % 64u == 0u);
    static_assert(
        sizeof(MoEOverlayDeviceControllerEconomyHeader) == 128u);
    static_assert(
        sizeof(MoEOverlayDeviceControllerMigrationCost) == 16u);
    static_assert(
        sizeof(MoEOverlayDeviceControllerFabricLayoutHeader) % 64u == 0u);
    static_assert(
        sizeof(MoEOverlayDeviceControllerParticipantMetadata) == 64u);
    static_assert(
        sizeof(MoEOverlayDeviceControllerParticipantRecord) == 128u);
    static_assert(
        sizeof(MoEOverlayDeviceControllerTransportRecord) == 128u);
    static_assert(
        sizeof(MoEOverlayDeviceControllerFabricGroupLayout) % 64u == 0u);
} // namespace llaminar2
