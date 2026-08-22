/**
 * @file MoEOverlayInferenceTransaction.h
 * @brief Authenticated fixed-slot scheduling protocol for ExpertOverlay inference.
 *
 * Heterogeneous ExpertOverlay execution cannot place a CUDA continuation graph,
 * ROCm expert graphs, and their MPI boundary inside one native device graph.  The
 * host is therefore permitted to schedule retained captured transactions at that
 * explicit heterogeneous boundary, but it must not become a second owner of
 * tokens, sampler state, KV positions, routing state, or model outputs.
 *
 * This file defines the narrow immutable ticket used at that boundary and a
 * device-free state machine that owns its fixed transaction slots.  A ticket
 * names one complete sparse graph transaction, not one transformer layer.  The
 * data plane continues to carry compact rows through the rank-batch transport;
 * the ticket only authenticates which already-built graph family may consume
 * them.  CPU unit tests use the same state machine as production so stale
 * generations, divergent placement epochs, premature slot reuse, and malformed
 * MTP geometry fail before device work is submitted.
 */

#pragma once

#include "MoEOverlayActivationEpochABI.h"
#include "MoEOverlayInferenceInterferenceProbe.h"

#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>
#include <string>
#include <type_traits>
#include <vector>

namespace llaminar2
{
    class MoEExpertOwnerMap;
    class IModelLoader;

    /** @brief Operation requested by one immutable transaction ticket. */
    enum class MoEOverlayInferenceTransactionAction : std::uint32_t
    {
        Execute = 1, ///< Submit exactly one retained sparse graph transaction.
        Complete = 2, ///< Close the current outer inference command successfully.
        Abort = 3, ///< Close the command fatally with a positive error code.
    };

    /**
     * @brief Fixed command-ring capacity for one retained MTP graph family.
     *
     * A command can contain one main condition graph, every admitted sidecar,
     * and one grouped verifier. Four remains the minimum for non-MTP and depth
     * two campaigns; larger configured depths receive capacity during setup.
     */
    [[nodiscard]] inline constexpr std::size_t
    moeOverlayInferenceTransactionSlotCount(int max_mtp_draft_depth) noexcept
    {
        if (max_mtp_draft_depth < 0)
            return 0;
        const std::size_t required =
            static_cast<std::size_t>(max_mtp_draft_depth) + 2u;
        return required < 4u ? 4u : required;
    }

    /**
     * @brief Immutable model-lifetime identity shared by both protocol endpoints.
     *
     * The two fingerprint lanes cover the resolved participant/domain/tier
     * topology.  `workspace_generation` covers every graph-stable pointer and
     * physical row bucket embedded by the retained graph family.  A recapture or
     * placement-plan rebuild creates a new identity and therefore a new protocol
     * owner instead of silently accepting a vaguely compatible ticket.
     */
    struct MoEOverlayInferenceTopologyIdentity
    {
        std::uint64_t workspace_generation = 0;
        std::uint64_t topology_fingerprint_low = 0;
        std::uint64_t topology_fingerprint_high = 0;
        std::int32_t source_world_rank = -1;
        std::int32_t target_world_rank = -1;

        /** @return Whether every graph/topology identity field is present. */
        [[nodiscard]] bool valid() const noexcept;

        bool operator==(
            const MoEOverlayInferenceTopologyIdentity &) const = default;
    };

    /**
     * @brief Graph-stable geometry mixed into a follower topology identity.
     *
     * The descriptor contains no request state or pointer values. A graph
     * rebuild increments `graph_family_generation`; a topology or capacity
     * change alters the fingerprint lanes. Both ranks independently derive the
     * same value from their frozen owner-map participant descriptors.
     */
    struct MoEOverlayInferenceGraphFamilyIdentity
    {
        std::uint64_t graph_family_generation = 0;
        int main_layer_count = 0;
        std::vector<int> mtp_source_layers;
        int max_graph_rows = 0;
        int max_decode_rows = 0;
        int max_request_count = 0;
        int max_mtp_draft_depth = 0;

        /** @return Whether all retained-family bounds are self-consistent. */
        [[nodiscard]] bool valid() const noexcept;
    };

    /**
     * @brief Resolve retained main/NextN graph geometry from model metadata.
     *
     * Both continuation and follower ranks call this exact function so the
     * topology fingerprint cannot drift through duplicated MTP-manifest logic.
     * Only routed-MoE sidecar source layers enter the returned graph family.
     */
    [[nodiscard]] MoEOverlayInferenceGraphFamilyIdentity
    resolveMoEOverlayInferenceGraphFamilyIdentity(
        const IModelLoader &loader,
        const std::string &architecture,
        int raw_layer_count,
        bool mtp_enabled,
        std::uint64_t graph_family_generation,
        int max_graph_rows,
        int max_decode_rows,
        int max_request_count,
        int max_mtp_draft_depth);

    /**
     * @brief Derive a pointer-independent rank-pair transaction identity.
     * @param owner_map Frozen participant topology shared by all ranks.
     * @param graph_family Exact retained graph-family geometry.
     * @param source_world_rank Continuation authority rank.
     * @param target_world_rank Remote follower rank.
     * @return Valid 128-bit topology identity and graph generation.
     * @throws std::invalid_argument for invalid geometry or rank membership.
     */
    [[nodiscard]] MoEOverlayInferenceTopologyIdentity
    makeMoEOverlayInferenceTopologyIdentity(
        const MoEExpertOwnerMap &owner_map,
        const MoEOverlayInferenceGraphFamilyIdentity &graph_family,
        int source_world_rank,
        int target_world_rank);

    /**
     * @brief Dynamic identity of one root-published outer inference command.
     *
     * `request_generation` changes on request reset. `command_id` increases for
     * every prefill/decode command inside that request. `initial_placement_epoch`
     * is the first RCU residency epoch the command may acquire; later execution
     * tickets may name a newer published epoch, but never an older one.
     */
    struct MoEOverlayInferenceCommandIdentity
    {
        std::uint64_t request_generation = 0;
        std::uint64_t command_id = 0;
        std::uint64_t initial_placement_epoch = 0;

        /** @return Whether this identity can open a transaction sequence. */
        [[nodiscard]] bool valid() const noexcept;

        bool operator==(
            const MoEOverlayInferenceCommandIdentity &) const = default;
    };

    /**
     * @brief Fixed-layout scheduler ticket for one complete sparse transaction.
     *
     * The ticket is intentionally a plain value suitable for a fixed registered
     * MPI slot. It contains scheduling identity and captured row geometry only.
     * In particular, it never contains tokens, KV positions, router outputs,
     * logits, sampler state, mutable depth-controller state, or response data.
     * Those values remain owned by the continuation device and sparse data plane.
     */
    struct MoEOverlayInferenceTransactionTicket
    {
        static constexpr std::uint32_t kMagic = 0x54494F4Du; // "MOIT"
        static constexpr std::uint32_t kABIVersion = 3u;

        std::uint32_t magic = kMagic;
        std::uint32_t abi_version = kABIVersion;
        MoEOverlayInferenceTransactionAction action =
            MoEOverlayInferenceTransactionAction::Execute;
        MoEOverlayInferenceGraphRole graph_role =
            MoEOverlayInferenceGraphRole::None;
        std::uint64_t request_generation = 0;
        std::uint64_t command_id = 0;
        std::uint64_t transaction_ordinal = 0;
        /** Absolute logical operation offset stamped into sparse wire keys. */
        std::uint64_t logical_step_id = 0;
        std::uint64_t workspace_generation = 0;
        /**
         * Scheduler-observed minimum placement epoch for this transaction.
         *
         * A host-authoritative topology uses this value exactly. A device-
         * authoritative topology may acquire a newer already-published epoch
         * at graph admission; the activation packet ABI authenticates that
         * exact endpoint-local ticket against this monotonic floor.
         */
        std::uint64_t placement_epoch = 0;
        std::uint64_t topology_fingerprint_low = 0;
        std::uint64_t topology_fingerprint_high = 0;
        std::int32_t source_world_rank = -1;
        std::int32_t target_world_rank = -1;
        std::int32_t request_count = 0;
        std::int32_t logical_rows_per_request = 0;
        std::int32_t physical_rows_per_request = 0;
        std::int32_t draft_depth = -1;
        std::int32_t sidecar_depth = -1;
        std::int32_t error_code = 0;
        /**
         * Complete caller-visible prefill schedule containing this ticket.
         *
         * Heterogeneous prefill may require many retained graph tickets.  The
         * calibration probe must time that logical schedule as one interval;
         * a physical migration can legitimately outlive any single segment.
         * Zero in every field means this ticket is not part of an aggregate
         * prefill schedule.  The fixed fields deliberately carry geometry,
         * never token values or mutable inference state.
         */
        std::int32_t prefill_schedule_real_rows = 0;
        std::int32_t prefill_schedule_execution_rows = 0;
        std::int32_t prefill_schedule_transaction_count = 0;
        std::int32_t prefill_schedule_reserved = 0;
        std::uint64_t prefill_schedule_fingerprint = 0;

        /** @return Model-lifetime topology identity carried by this ticket. */
        [[nodiscard]] MoEOverlayInferenceTopologyIdentity topologyIdentity()
            const noexcept;

        /** @return Dynamic outer-command identity carried by this ticket. */
        [[nodiscard]] MoEOverlayInferenceCommandIdentity commandIdentity()
            const noexcept;

        /**
         * @brief Reconstruct the aggregate prefill workload carried on wire.
         * @return Valid complete schedule identity, or an invalid identity when
         *         this ticket is not part of an aggregate prefill schedule.
         */
        [[nodiscard]] MoEOverlayInferenceWorkloadIdentity
        prefillScheduleWorkload() const noexcept;

        /** @return Whether ABI and role-specific geometry are self-consistent. */
        [[nodiscard]] bool valid() const noexcept;

        /** @return Stable diagnostic text containing every scheduling field. */
        [[nodiscard]] std::string toString() const;

        bool operator==(
            const MoEOverlayInferenceTransactionTicket &) const = default;
    };

    static_assert(
        std::is_trivially_copyable_v<MoEOverlayInferenceTransactionTicket>);
    static_assert(sizeof(MoEOverlayInferenceTransactionTicket) == 136u);

    /**
     * @brief Build and validate one retained-graph execution ticket.
     *
     * @throws std::invalid_argument when any identity or role geometry is
     *         invalid. Construction failure is a setup/controller defect and is
     *         never repaired by choosing another graph family.
     */
    [[nodiscard]] MoEOverlayInferenceTransactionTicket
    makeMoEOverlayInferenceExecutionTicket(
        const MoEOverlayInferenceTopologyIdentity &topology,
        const MoEOverlayInferenceCommandIdentity &command,
        std::uint64_t transaction_ordinal,
        std::uint64_t logical_step_id,
        std::uint64_t placement_epoch,
        MoEOverlayInferenceGraphRole graph_role,
        int request_count,
        int logical_rows_per_request,
        int physical_rows_per_request,
        int draft_depth = -1,
        int sidecar_depth = -1,
        const MoEOverlayInferenceWorkloadIdentity &
            prefill_schedule_workload = {});

    /**
     * @brief Build a successful or fatal terminal ticket for one command.
     *
     * @param action Complete or Abort.
     * @param error_code Zero for Complete; positive for Abort.
     * @throws std::invalid_argument for an execution action or inconsistent code.
     */
    [[nodiscard]] MoEOverlayInferenceTransactionTicket
    makeMoEOverlayInferenceTerminalTicket(
        const MoEOverlayInferenceTopologyIdentity &topology,
        const MoEOverlayInferenceCommandIdentity &command,
        std::uint64_t transaction_ordinal,
        std::uint64_t placement_epoch,
        MoEOverlayInferenceTransactionAction action,
        int error_code = 0);

    /** @brief Lifecycle of one fixed transaction ring slot. */
    enum class MoEOverlayInferenceTransactionSlotState : std::uint8_t
    {
        Available = 0, ///< Slot has no live MPI/device owner.
        Accepted, ///< Ticket authenticated; graph submission has not begun.
        Submitted, ///< Retained graph and its data-plane work are in flight.
        ReturnReady, ///< Result is published and the return send may be retired.
    };

    /** @brief Lifecycle of one outer command at a protocol endpoint. */
    enum class MoEOverlayInferenceProtocolState : std::uint8_t
    {
        Idle = 0,
        Active,
        Complete,
        Failed,
    };

    /** @brief Outcome of authenticating the next ticket in a command sequence. */
    enum class MoEOverlayInferenceAdmissionStatus : std::uint8_t
    {
        Accepted = 0, ///< Execution ticket owns the returned slot.
        Complete, ///< Successful terminal ticket closed the command.
        Aborted, ///< Fatal terminal ticket closed the command.
        Backpressured, ///< Exact next work is valid but its fixed slot is live.
        Rejected, ///< Ticket is malformed, stale, divergent, or out of order.
    };

    /** @brief Typed result returned by transaction admission. */
    struct MoEOverlayInferenceAdmission
    {
        static constexpr std::size_t kNoSlot =
            std::numeric_limits<std::size_t>::max();

        MoEOverlayInferenceAdmissionStatus status =
            MoEOverlayInferenceAdmissionStatus::Rejected;
        std::size_t slot_index = kNoSlot;
        std::string error;

        /** @return Whether an execution ticket acquired one fixed slot. */
        [[nodiscard]] bool accepted() const noexcept
        {
            return status == MoEOverlayInferenceAdmissionStatus::Accepted;
        }

        /** @return Whether this result is a successful command terminal. */
        [[nodiscard]] bool completed() const noexcept
        {
            return status == MoEOverlayInferenceAdmissionStatus::Complete;
        }
    };

    /**
     * @brief Device-free authority for a fixed ring of sparse transaction slots.
     *
     * One orchestration/progress owner calls these methods in order. The class
     * performs no I/O, device work, waits, or hot-path allocation. `accept()`
     * maps an exact monotonic ordinal to a fixed slot; that slot remains owned
     * until `markSubmitted()`, `markReturnReady()`, and `retire()` complete.
     * A wrapped ordinal therefore reports typed backpressure instead of allowing
     * an MPI request or captured graph to observe overwritten storage.
     */
    class MoEOverlayInferenceTransactionProtocol final
    {
    public:
        /** @brief Immutable topology, geometry ceilings, and fixed ring size. */
        struct Config
        {
            MoEOverlayInferenceTopologyIdentity topology;
            std::size_t slot_count = 0;
            int max_request_count = 0;
            int max_rows_per_request = 0;
            int max_mtp_draft_depth = 0;
        };

        /**
         * @brief Allocate the fixed CPU metadata ring and validate all ceilings.
         * @throws std::invalid_argument for an invalid topology or capacity.
         */
        explicit MoEOverlayInferenceTransactionProtocol(Config config);

        /**
         * @brief Open one root-published command at its initial placement epoch.
         *
         * Request generations must increase, or command ids must increase within
         * the current generation. A failed command is process-terminal and cannot
         * be reopened. A completed command may be followed by the next command.
         */
        bool beginCommand(
            const MoEOverlayInferenceCommandIdentity &command,
            std::string *error = nullptr);

        /**
         * @brief Authenticate the next ticket and claim its deterministic slot.
         *
         * Exact next work may return Backpressured while its wrapped slot remains
         * live; callers may progress/retire that slot and retry the same ticket.
         * All other non-accepted results are terminal protocol decisions.
         */
        [[nodiscard]] MoEOverlayInferenceAdmission accept(
            const MoEOverlayInferenceTransactionTicket &ticket);

        /** @brief Record that the accepted slot's graph/data work was submitted. */
        bool markSubmitted(
            std::size_t slot_index,
            const MoEOverlayInferenceTransactionTicket &ticket,
            std::string *error = nullptr);

        /** @brief Record that the exact slot's result is ready for return transport. */
        bool markReturnReady(
            std::size_t slot_index,
            const MoEOverlayInferenceTransactionTicket &ticket,
            std::string *error = nullptr);

        /** @brief Release a return-complete slot for a later wrapped ordinal. */
        bool retire(
            std::size_t slot_index,
            const MoEOverlayInferenceTransactionTicket &ticket,
            std::string *error = nullptr);

        /** @return Current outer-command lifecycle. */
        [[nodiscard]] MoEOverlayInferenceProtocolState state() const noexcept
        {
            return state_;
        }

        /** @return Ordinal required by the next execution or terminal ticket. */
        [[nodiscard]] std::uint64_t nextTransactionOrdinal() const noexcept
        {
            return next_transaction_ordinal_;
        }

        /** @return Number of fixed slots not yet retired. */
        [[nodiscard]] std::size_t inFlightSlotCount() const noexcept;

        /** @return Lifecycle of one fixed slot, or empty for an invalid index. */
        [[nodiscard]] std::optional<MoEOverlayInferenceTransactionSlotState>
        slotState(std::size_t slot_index) const noexcept;

        /** @return Most recently accepted placement epoch in this command. */
        [[nodiscard]] std::uint64_t currentPlacementEpoch() const noexcept
        {
            return current_placement_epoch_;
        }

        /** @return Immutable ring capacity allocated by the constructor. */
        [[nodiscard]] std::size_t slotCount() const noexcept
        {
            return slots_.size();
        }

        /** @return Immutable model/topology identity authenticated by this protocol. */
        [[nodiscard]] const MoEOverlayInferenceTopologyIdentity &
        topologyIdentity() const noexcept
        {
            return config_.topology;
        }

    private:
        /** @brief Fixed metadata owned by one registered transport slot. */
        struct Slot
        {
            MoEOverlayInferenceTransactionSlotState state =
                MoEOverlayInferenceTransactionSlotState::Available;
            MoEOverlayInferenceTransactionTicket ticket{};
        };

        /** @brief Validate an execution ticket against immutable/dynamic bounds. */
        [[nodiscard]] std::string validateExecutionTicket(
            const MoEOverlayInferenceTransactionTicket &ticket) const;

        /** @brief Perform one exact slot transition without accepting aliases. */
        bool transitionSlot(
            std::size_t slot_index,
            const MoEOverlayInferenceTransactionTicket &ticket,
            MoEOverlayInferenceTransactionSlotState expected,
            MoEOverlayInferenceTransactionSlotState next,
            std::string *error);

        Config config_;
        std::vector<Slot> slots_;
        MoEOverlayInferenceProtocolState state_ =
            MoEOverlayInferenceProtocolState::Idle;
        MoEOverlayInferenceCommandIdentity active_command_{};
        std::uint64_t last_request_generation_ = 0;
        std::uint64_t last_command_id_ = 0;
        std::uint64_t next_transaction_ordinal_ = 1;
        std::uint64_t current_placement_epoch_ = 0;
    };

} // namespace llaminar2
