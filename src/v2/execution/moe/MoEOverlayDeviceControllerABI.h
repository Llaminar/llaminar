/**
 * @file MoEOverlayDeviceControllerABI.h
 * @brief Fixed-width shared ABI for a topology-wide GPU ExpertOverlay authority.
 *
 * One continuation-root GPU owns policy for Static, Dynamic, and current-batch
 * LLEP. Group-root followers exchange only immutable transaction records and
 * monotonic acknowledgements through device-visible node-local pages. Records
 * contain no host-authored routing decision and require no fixed tier count:
 * the fabric allocates one group record per resolved controller group.
 */

#pragma once

#include <cstdint>
#include <type_traits>

namespace llaminar2
{
    /** Binary identity of the topology-wide controller ABI (`MOEC`). */
    inline constexpr std::uint32_t kMoEOverlayDeviceControllerMagic =
        0x43454f4du;

    /** Version of every fixed-width controller record in this header. */
    inline constexpr std::uint32_t kMoEOverlayDeviceControllerVersion = 8u;

    /** Maximum participants represented by one node-local inference epoch barrier. */
    inline constexpr std::uint32_t
        kMoEOverlayDeviceControllerInferenceEpochMaxParticipants = 8u;

    /** Binary identity of one continuation-transaction epoch record (`MOIT`). */
    inline constexpr std::uint32_t
        kMoEOverlayDeviceControllerInferenceEpochMagic = 0x54494f4du;

    /** Version of @ref MoEOverlayDeviceControllerInferenceEpochRecord. */
    inline constexpr std::uint32_t
        kMoEOverlayDeviceControllerInferenceEpochVersion = 1u;

    /**
     * @brief Device-owned exact epoch handoff for one symmetric continuation graph.
     *
     * A LocalTP continuation graph is valid only when every sibling acquires the
     * same topology-wide placement epoch. Sampling `admission_epoch` separately
     * on each GPU is racy: maintenance may publish between those samples. This
     * record turns admission into a two-phase, monotonically sequenced barrier.
     *
     * Each member first increments its participant-local RCU acquisition guard,
     * then release-publishes the next value in its disjoint `arrival_sequence`
     * lane. The immutable publisher waits until every bit in `participant_mask`
     * has arrived, snapshots the live admission epoch once, and release-publishes
     * `publication_sequence`. Followers acquire that sequence and therefore see
     * the exact `epoch`. Keeping the local guard raised while waiting prevents a
     * background retirement from reclaiming the selected bank on a late sibling.
     *
     * One record is sufficient for a serial request slot: a participant cannot
     * enter sequence N+1 until its request-local ticket from N has been released,
     * and the publisher cannot overwrite the record until every member has
     * arrived at N+1. The mapped record is node-local by construction and must
     * never be used as a cross-node transport.
     */
    struct alignas(64) MoEOverlayDeviceControllerInferenceEpochRecord
    {
        std::uint32_t magic =
            kMoEOverlayDeviceControllerInferenceEpochMagic;
        std::uint32_t version =
            kMoEOverlayDeviceControllerInferenceEpochVersion;
        /** Dense global participant-id bits belonging to continuation. */
        std::uint32_t participant_mask = 0u;
        /** Sole participant allowed to write epoch/publication sequence. */
        std::uint32_t publisher_participant_id = 0u;
        /** Frozen topology identity shared with the surrounding controller. */
        std::uint64_t topology_fingerprint = 0u;
        /** Exact topology-wide admission chosen for `publication_sequence`. */
        std::uint64_t epoch = 0u;
        /** Monotonic release edge authenticating @ref epoch. */
        std::uint64_t publication_sequence = 0u;
        /** Disjoint system-release arrival lane indexed by global participant id. */
        std::uint64_t arrival_sequence
            [kMoEOverlayDeviceControllerInferenceEpochMaxParticipants] = {};
        std::uint64_t reserved[3] = {};
    };

    /** Sole policy operation represented by one controller transaction. */
    enum class MoEOverlayDeviceControllerTransactionKind : std::uint32_t
    {
        Invalid = 0u,
        StaticCheck = 1u,       ///< Prove immobility without changing placement.
        DynamicPlacement = 2u,  ///< Publish one durable `E -> E+1` placement.
        CurrentBatchLLEP = 3u,  ///< Publish and later restore transient assignment.
        /** Restore the loader-prepared owner table before context reuse. */
        PreparedContextRestore = 4u,
    };

    /** Scheduler boundary triggering a phase-complete Dynamic demand snapshot. */
    enum class MoEOverlayDeviceDemandPhase : std::uint32_t
    {
        Invalid = 0u,
        Prefill = 1u,
        Decode = 2u,
    };

    /** Global leader-owned transaction lifecycle. */
    enum class MoEOverlayDeviceControllerState : std::uint32_t
    {
        Uninitialized = 0u,
        Idle = 1u,
        CollectingSnapshots = 2u,
        PreparingFollowers = 3u,
        PublishingFollowers = 4u,
        Admitted = 5u,
        RetiringDurableEpoch = 6u,
        RestoringLLEP = 7u,
        Complete = 8u,
        Error = 9u,
    };

    /** First terminal protocol failure retained in shared device state. */
    enum class MoEOverlayDeviceControllerError : std::uint32_t
    {
        None = 0u,
        InvalidControl = 1u,
        InvalidState = 2u,
        InvalidTransaction = 3u,
        InvalidGroup = 4u,
        InvalidTopology = 5u,
        InvalidCommand = 6u,
        EpochOverflow = 7u,
        ConflictingPublication = 8u,
        /** Collected ownership/residency words cannot form one durable epoch. */
        InvalidSnapshot = 9u,
        /** The command-authenticated physical transport failed asynchronously. */
        PhysicalTransportFailure = 10u,
    };

    /** Physical action represented by one immutable overlay command entry. */
    enum class MoEOverlayDeviceMovementOp : std::uint32_t
    {
        Invalid = 0u,
        /** Move one complete expert into the next durable residency bank. */
        DurableMove = 1u,
        /** Materialize one request-local expert copy for current-batch LLEP. */
        TransientArrival = 2u,
        /** Assign rows to an already-resident request-local execution endpoint. */
        TransientAssignment = 3u,
    };

    /**
     * @brief Policy objective carried by one immutable device command.
     *
     * This is a wire-level spelling of the authority-owned movement axis. It
     * occupies the command's `flags` word so adding the semantic identity does
     * not change fixed arena geometry. The host follower authenticates and
     * translates it but never reconstructs it from endpoint priorities.
     */
    enum class MoEOverlayDeviceMovementAxis : std::uint32_t
    {
        TierResidency = 1u,
        ParticipantPlacement = 2u,
        Combined = 3u,
    };

    /** Per-entry binary identity (`MOEM`). */
    inline constexpr std::uint32_t kMoEOverlayDeviceMovementCommandMagic =
        0x4d454f4du;

    /** Version of @ref MoEOverlayDeviceMovementCommand. */
    inline constexpr std::uint32_t kMoEOverlayDeviceMovementCommandVersion =
        3u;

    /** Sentinel used when an action has no physical source or destination slot. */
    inline constexpr std::uint32_t kMoEOverlayDeviceInvalidSlot =
        0xffffffffu;

    /**
     * @brief One conflict-checkable movement or assignment in a parallel epoch.
     *
     * Entries are written by the sole device policy leader in canonical
     * destination order. Movement entries use `ordinal` as their globally
     * unique payload lane, making payload aliasing unrepresentable without
     * adding a second serialized wave. Physical source lookup and destination
     * slot reservation deliberately do not appear in this topology-wide ABI:
     * each participant resolves its current bank and reserves one inactive
     * local slot during the prepare phase. That keeps raw CUDA/ROCm addresses
     * and backend-local directory generations inside their owning device.
     * A prepare acknowledgement is legal only after those local reservations
     * are complete and conflict-free.
     */
    struct alignas(64) MoEOverlayDeviceMovementCommand
    {
        std::uint32_t magic = kMoEOverlayDeviceMovementCommandMagic;
        std::uint32_t version = kMoEOverlayDeviceMovementCommandVersion;
        std::uint32_t op = static_cast<std::uint32_t>(
            MoEOverlayDeviceMovementOp::Invalid);
        std::uint32_t ordinal = 0u;
        std::uint32_t layer = 0u;
        std::uint32_t expert = 0u;
        std::uint32_t source_participant = 0u;
        std::uint32_t destination_participant = 0u;
        std::uint32_t payload_slot = kMoEOverlayDeviceInvalidSlot;
        /** Encoded @ref MoEOverlayDeviceMovementAxis authored by policy. */
        std::uint32_t flags = 0u;
        std::uint64_t payload_bytes = 0u;
        /** Durable source epoch, or pinned durable epoch for transient LLEP. */
        std::uint64_t source_epoch = 0u;
        /** Candidate durable epoch; unchanged for request-scoped LLEP. */
        std::uint64_t candidate_epoch = 0u;
    };

    /**
     * @brief Topology-wide leader state visible to every controller group.
     *
     * Mutable ownership is strict: only the leader GPU writes lifecycle,
     * transaction, command, commit, admission, and error fields. Followers
     * write only their own @ref MoEOverlayDeviceControllerGroupRecord. GPU
     * implementations use system-scope acquire/release atomics for every
     * monotonic publication field.
     */
    struct alignas(64) MoEOverlayDeviceControllerSharedHeader
    {
        std::uint32_t magic = kMoEOverlayDeviceControllerMagic;
        std::uint32_t version = kMoEOverlayDeviceControllerVersion;
        std::uint32_t group_count = 0u;
        std::uint32_t leader_group_id = 0u;
        std::uint32_t leader_participant_id = 0u;
        std::uint32_t state = static_cast<std::uint32_t>(
            MoEOverlayDeviceControllerState::Uninitialized);
        std::uint32_t transaction_kind = static_cast<std::uint32_t>(
            MoEOverlayDeviceControllerTransactionKind::Invalid);
        std::uint32_t error_code = static_cast<std::uint32_t>(
            MoEOverlayDeviceControllerError::None);
        std::uint64_t topology_fingerprint = 0u;
        std::uint64_t current_durable_epoch = 0u;
        std::uint64_t transaction_id = 0u;
        std::uint64_t base_epoch = 0u;
        std::uint64_t candidate_epoch = 0u;
        /**
         * Latest sealed command. Opening the next snapshot does not clear this
         * receipt or its payload: a delayed transport worker may still need an
         * empty completed command. The next snapshot fan-in is the reuse edge.
         */
        std::uint64_t command_transaction = 0u;
        std::uint64_t commit_transaction = 0u;
        std::uint64_t admission_transaction = 0u;
        std::uint64_t admission_epoch = 0u;
        std::uint64_t active_llep_transaction = 0u;
        /**
         * Latest transaction whose terminal effects are globally visible.
         *
         * Unlike `state`, this release-published receipt is never cleared when
         * the leader starts a later transaction. Independently paced rank-local
         * schedulers can therefore certify transaction N after the authority
         * has already entered N+1 without retaining a host-side epoch mirror.
         */
        std::uint64_t completed_transaction = 0u;
        std::uint32_t error_group_id = 0xffffffffu;
        /**
         * First model layer considered by the next Dynamic policy wave.
         *
         * The device authority alone advances this round-robin cursor. It
         * prevents a bounded cycle budget from permanently starving higher
         * model layers without imposing any topology-specific layer count.
         */
        std::uint32_t placement_layer_cursor = 0u;
        /**
         * Phase intent for transaction_id, published with its open ticket.
         * This is separate from the retained command's demand_phase because
         * snapshot N+1 can coexist with a not-yet-acquired empty command N.
         */
        std::uint32_t transaction_demand_phase = static_cast<std::uint32_t>(
            MoEOverlayDeviceDemandPhase::Invalid);
    };

    /**
     * @brief One group-root publication and acknowledgement lane.
     *
     * Payload arrays are fabric-owned variable regions. Digests bind those
     * bytes to the monotonic transaction words below, allowing the leader and
     * followers to reject stale or partially overwritten payloads without a
     * host mirror.
     */
    struct alignas(64) MoEOverlayDeviceControllerGroupRecord
    {
        std::uint32_t magic = kMoEOverlayDeviceControllerMagic;
        std::uint32_t version = kMoEOverlayDeviceControllerVersion;
        std::uint32_t group_id = 0u;
        std::uint32_t root_participant_id = 0u;
        std::uint64_t topology_fingerprint = 0u;
        std::uint64_t snapshot_digest = 0u;
        std::uint64_t snapshot_observations = 0u;
        std::uint64_t snapshot_transaction = 0u;
        std::uint64_t prepared_transaction = 0u;
        std::uint64_t published_transaction = 0u;
        std::uint64_t restored_transaction = 0u;
        std::uint64_t retired_epoch = 0u;
        std::uint32_t status_code = static_cast<std::uint32_t>(
            MoEOverlayDeviceControllerError::None);
        std::uint32_t reserved = 0u;
    };

    /**
     * @brief Immutable leader-authored command identity for one transaction.
     *
     * `command_count` covers separately allocated fixed-width movement or
     * assignment entries. `packed_weight_bytes` is zero for Static and may be
     * zero for resident-only LLEP assignment; a parity cell that claims weight
     * movement imposes the stronger positive-byte evidence requirement.
     */
    struct alignas(64) MoEOverlayDeviceControllerCommandHeader
    {
        std::uint32_t magic = kMoEOverlayDeviceControllerMagic;
        std::uint32_t version = kMoEOverlayDeviceControllerVersion;
        std::uint32_t kind = static_cast<std::uint32_t>(
            MoEOverlayDeviceControllerTransactionKind::Invalid);
        std::uint32_t command_count = 0u;
        std::uint64_t topology_fingerprint = 0u;
        std::uint64_t transaction_id = 0u;
        std::uint64_t base_epoch = 0u;
        std::uint64_t candidate_epoch = 0u;
        std::uint64_t command_digest = 0u;
        std::uint64_t packed_weight_bytes = 0u;
        /** Number of commands admitted into the one concurrent fan-out. */
        std::uint32_t parallel_command_count = 0u;
        /** Exactly one for movement, zero for a Static no-movement check. */
        std::uint32_t movement_round_count = 0u;
        /** Must remain zero; conflicting destinations are never serialized. */
        std::uint32_t hazard_count = 0u;
        /** Sealed phase copied from the matching transaction's open intent. */
        std::uint32_t demand_phase = static_cast<std::uint32_t>(
            MoEOverlayDeviceDemandPhase::Invalid);
        /** Total routed observations in the exact device-authored snapshot. */
        std::uint64_t snapshot_observations = 0u;
        /** Cross-tier priority-rank cost before the admitted wave. */
        std::uint64_t priority_cost_before = 0u;
        /** Cross-tier priority-rank cost after the admitted wave. */
        std::uint64_t priority_cost_after = 0u;
        /** Sum of per-priority participant maxima before the wave. */
        std::uint64_t same_priority_makespan_before = 0u;
        /** Sum of per-priority participant maxima after the wave. */
        std::uint64_t same_priority_makespan_after = 0u;
        /** Complete capacity-preserving cycles admitted by device policy. */
        std::uint32_t accepted_cycles = 0u;
        /** Candidate cycles rejected by the device economy gates. */
        std::uint32_t rejected_cycles = 0u;
        /** Device-authored count of lower-integer-priority movements. */
        std::uint32_t promotions = 0u;
        /** Device-authored count of higher-integer-priority movements. */
        std::uint32_t demotions = 0u;
        /** Device-authored count of equal-priority movements. */
        std::uint32_t same_priority_moves = 0u;
        /** Number of layers changed by the bounded wave. */
        std::uint32_t changed_layers = 0u;
        /** First layer examined by the fair cyclic scan for this command. */
        std::uint32_t layer_scan_start = 0u;
        /** Device-owned cursor to use for the next Dynamic transaction. */
        std::uint32_t layer_scan_next = 0u;
        /** Token-horizon service time saved by admitted measured-economy cycles. */
        std::uint64_t projected_service_gain_ns = 0u;
        /** Certified transfer and repack critical-path charge. */
        std::uint64_t projected_transfer_and_repack_ns = 0u;
        /** Certified inference slowdown while maintenance overlaps. */
        std::uint64_t projected_inference_interference_ns = 0u;
        /** Exact positive remainder after both measured movement charges. */
        std::uint64_t projected_net_benefit_ns = 0u;
        /** Candidate cycles rejected because payoff was uneconomical. */
        std::uint32_t payoff_rejected_cycles = 0u;
        /** Candidate cycles rejected by committed movement hysteresis. */
        std::uint32_t residency_rejected_cycles = 0u;
    };

    static_assert(
        std::is_trivially_copyable_v<
            MoEOverlayDeviceControllerSharedHeader>);
    static_assert(
        std::is_trivially_copyable_v<
            MoEOverlayDeviceControllerGroupRecord>);
    static_assert(
        std::is_trivially_copyable_v<
            MoEOverlayDeviceControllerCommandHeader>);
    static_assert(
        std::is_trivially_copyable_v<MoEOverlayDeviceMovementCommand>);
    static_assert(
        std::is_trivially_copyable_v<
            MoEOverlayDeviceControllerInferenceEpochRecord>);
    static_assert(
        alignof(MoEOverlayDeviceControllerSharedHeader) == 64u);
    static_assert(
        alignof(MoEOverlayDeviceControllerGroupRecord) == 64u);
    static_assert(
        alignof(MoEOverlayDeviceControllerCommandHeader) == 64u);
    static_assert(alignof(MoEOverlayDeviceMovementCommand) == 64u);
    static_assert(
        alignof(MoEOverlayDeviceControllerInferenceEpochRecord) == 64u);
    static_assert(
        sizeof(MoEOverlayDeviceControllerSharedHeader) == 192u,
        "The CUDA/HIP mapped controller header ABI must remain exactly three cache lines");
    static_assert(
        sizeof(MoEOverlayDeviceControllerGroupRecord) == 128u,
        "Each independently published group record must occupy two cache lines");
    static_assert(
        sizeof(MoEOverlayDeviceControllerCommandHeader) == 192u,
        "The immutable controller command ABI must remain exactly three cache lines");
    static_assert(
        sizeof(MoEOverlayDeviceMovementCommand) == 64u,
        "Each parallel movement command must occupy exactly one cache line");
    static_assert(
        sizeof(MoEOverlayDeviceControllerInferenceEpochRecord) == 128u,
        "The continuation admission barrier must occupy two cache lines");
} // namespace llaminar2
