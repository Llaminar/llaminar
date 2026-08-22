/**
 * @file MoEOverlayDeviceControllerProtocol.h
 * @brief CPU executable specification for topology-wide device controller RCU.
 *
 * The production CUDA/HIP kernels operate on the same ABI with system-scope
 * atomics. This class is the CPU backend/reference state machine and adversarial
 * unit-test oracle. It must never be used to shadow or repair a GPU-owned live
 * controller: an all-GPU runner embeds the records in device-visible shared
 * pages and only device kernels mutate them during inference.
 */

#pragma once

#include "MoEOverlayDeviceControllerABI.h"

#include <cstdint>
#include <optional>
#include <span>

namespace llaminar2
{
    /** Test-oracle evidence copied before a host-simulated command release. */
    struct MoEOverlayDeviceControllerCommandEvidence
    {
        std::uint64_t snapshot_observations = 0u;
        std::uint64_t priority_cost_before = 0u;
        std::uint64_t priority_cost_after = 0u;
        std::uint64_t same_priority_makespan_before = 0u;
        std::uint64_t same_priority_makespan_after = 0u;
        std::uint32_t accepted_cycles = 0u;
        std::uint32_t rejected_cycles = 0u;
        std::uint32_t promotions = 0u;
        std::uint32_t demotions = 0u;
        std::uint32_t same_priority_moves = 0u;
        std::uint32_t changed_layers = 0u;
        std::uint32_t layer_scan_start = 0u;
        std::uint32_t layer_scan_next = 0u;
        /** Token-horizon service saving admitted by measured policy. */
        std::uint64_t projected_service_gain_ns = 0u;
        /** Certified transfer/repack charge for admitted cycles. */
        std::uint64_t projected_transfer_and_repack_ns = 0u;
        /** Certified overlap interference charge for admitted cycles. */
        std::uint64_t projected_inference_interference_ns = 0u;
        /** Positive measured benefit retained after every admitted cost. */
        std::uint64_t projected_net_benefit_ns = 0u;
        /** Cycles rejected because their measured payoff was insufficient. */
        std::uint32_t payoff_rejected_cycles = 0u;
        /** Cycles rejected by committed-movement hysteresis. */
        std::uint32_t residency_rejected_cycles = 0u;
    };

    /**
     * @brief Lock-free reference protocol for one leader and variable followers.
     *
     * Leader methods serialize policy transitions. Group publication methods
     * may run concurrently because each group owns a disjoint record. Every
     * payload digest is stored before its release transaction word; consumers
     * acquire that word before inspecting payload metadata.
     */
    class MoEOverlayDeviceControllerProtocol final
    {
    public:
        /**
         * @brief Initialize pristine shared records before any graph is built.
         * @param header Shared topology-wide header.
         * @param groups One record per dense controller group id.
         * @param command Shared leader-authored command header.
         * @param topology_fingerprint Non-zero immutable topology identity.
         * @param leader_group_id Dense group id containing the sole leader.
         * @param leader_participant_id Stable global participant id.
         * @param initial_durable_epoch Positive first executable placement.
         * @param group_root_participants Root participant indexed by group id.
         * @throws std::invalid_argument for malformed immutable identity.
         * @throws std::logic_error when records were already initialized.
         */
        static void initialize(
            MoEOverlayDeviceControllerSharedHeader &header,
            std::span<MoEOverlayDeviceControllerGroupRecord> groups,
            MoEOverlayDeviceControllerCommandHeader &command,
            std::uint64_t topology_fingerprint,
            std::uint32_t leader_group_id,
            std::uint32_t leader_participant_id,
            std::uint64_t initial_durable_epoch,
            std::span<const std::uint32_t> group_root_participants);

        /** Bind the protocol to one authoritative shared record family. */
        MoEOverlayDeviceControllerProtocol(
            MoEOverlayDeviceControllerSharedHeader &header,
            std::span<MoEOverlayDeviceControllerGroupRecord> groups,
            MoEOverlayDeviceControllerCommandHeader &command) noexcept;

        /**
         * @brief Begin one sole-authority transaction on the leader.
         * @param kind Static, durable Dynamic, or transient current-batch LLEP.
         * @param phase Prefill/decode for Dynamic; Invalid for phase-free kinds.
         * @return New positive transaction id, or empty after a fatal violation.
         */
        [[nodiscard]] std::optional<std::uint64_t> beginTransaction(
            MoEOverlayDeviceControllerTransactionKind kind,
            MoEOverlayDeviceDemandPhase phase) noexcept;

        /**
         * @brief Publish one group-root histogram/directory snapshot.
         * @param group_id Exact dense group id owned by the caller.
         * @param transaction_id Active transaction.
         * @param snapshot_digest Non-zero digest of immutable snapshot payload.
         * @param observations Number of routed observations represented.
         * @return False and poison the transaction on identity/state conflict.
         */
        [[nodiscard]] bool publishGroupSnapshot(
            std::uint32_t group_id,
            std::uint64_t transaction_id,
            std::uint64_t snapshot_digest,
            std::uint64_t observations) noexcept;

        /** @return Whether every group snapshot is acquired for @p transaction_id. */
        [[nodiscard]] bool allSnapshotsReady(
            std::uint64_t transaction_id) const noexcept;

        /**
         * @brief Publish the sole leader-authored command after snapshot fan-in.
         * @param transaction_id Active transaction.
         * @param command_count Number of separately stored plan entries.
         * @param command_digest Non-zero digest covering every plan entry.
         * @param packed_weight_bytes Exact weight bytes the command will move.
         * @param parallel_command_count Commands in the one concurrent fan-out.
         * @param movement_round_count One for movement, zero for Static.
         * @param hazard_count Conflicting commands; must be zero.
         * @param evidence Optional CPU-oracle evidence for device-free tests.
         * @return False after a fatal state, identity, or mode violation.
         */
        [[nodiscard]] bool publishCommand(
            std::uint64_t transaction_id,
            std::uint32_t command_count,
            std::uint64_t command_digest,
            std::uint64_t packed_weight_bytes,
            std::uint32_t parallel_command_count,
            std::uint32_t movement_round_count,
            std::uint32_t hazard_count,
            const MoEOverlayDeviceControllerCommandEvidence &evidence = {})
            noexcept;

        /** Publish one follower group's completed prepare/weight-ready edge. */
        [[nodiscard]] bool acknowledgePrepared(
            std::uint32_t group_id,
            std::uint64_t transaction_id) noexcept;

        /** @return Whether every group has prepared @p transaction_id. */
        [[nodiscard]] bool allGroupsPrepared(
            std::uint64_t transaction_id) const noexcept;

        /** Publish the leader commit edge after every follower is prepared. */
        [[nodiscard]] bool beginCommit(
            std::uint64_t transaction_id) noexcept;

        /** Publish one follower group's local inactive-bank/assignment switch. */
        [[nodiscard]] bool acknowledgePublished(
            std::uint32_t group_id,
            std::uint64_t transaction_id) noexcept;

        /** @return Whether every group published the named transaction locally. */
        [[nodiscard]] bool allGroupsPublished(
            std::uint64_t transaction_id) const noexcept;

        /**
         * @brief Open topology-wide admission after every local publication.
         * @return False when any group is missing or the command is stale.
         *
         * Dynamic advances the durable epoch. Static completes without changing
         * it. LLEP opens a transient assignment lease that must be restored.
         */
        [[nodiscard]] bool publishAdmission(
            std::uint64_t transaction_id) noexcept;

        /** Begin the Dynamic grace-period retirement fan-out. */
        [[nodiscard]] bool beginDynamicRetirement(
            std::uint64_t transaction_id) noexcept;

        /** Acknowledge that one group reclaimed the exact prior durable epoch. */
        [[nodiscard]] bool acknowledgeRetired(
            std::uint32_t group_id,
            std::uint64_t transaction_id,
            std::uint64_t retired_epoch) noexcept;

        /** Complete Dynamic only after every old local bank is reclaimed. */
        [[nodiscard]] bool completeDynamicRetirement(
            std::uint64_t transaction_id) noexcept;

        /** Begin restoration of one admitted current-batch LLEP assignment. */
        [[nodiscard]] bool beginLLEPRestore(
            std::uint64_t transaction_id) noexcept;

        /** Acknowledge one group's owner-only assignment restoration. */
        [[nodiscard]] bool acknowledgeLLEPRestored(
            std::uint32_t group_id,
            std::uint64_t transaction_id) noexcept;

        /** Complete LLEP only after every transient group state is restored. */
        [[nodiscard]] bool completeLLEPRestore(
            std::uint64_t transaction_id) noexcept;

        /** @return Current global state observed with acquire ordering. */
        [[nodiscard]] MoEOverlayDeviceControllerState state() const noexcept;

        /** @return First retained terminal error. */
        [[nodiscard]] MoEOverlayDeviceControllerError error() const noexcept;

        /** @return Exact current durable placement epoch. */
        [[nodiscard]] std::uint64_t currentDurableEpoch() const noexcept;

        /** @return Exact transaction currently admitted for LLEP, or zero. */
        [[nodiscard]] std::uint64_t activeLLEPTransaction() const noexcept;

        /** @return Whether immutable ABI and topology identity remain valid. */
        [[nodiscard]] bool valid() const noexcept;

    private:
        /** Publish the first fatal protocol error and enter terminal Error. */
        bool fail(
            MoEOverlayDeviceControllerError error,
            std::uint32_t group_id = 0xffffffffu) noexcept;

        /** Resolve a dense group id or poison the shared transaction. */
        MoEOverlayDeviceControllerGroupRecord *requireGroup(
            std::uint32_t group_id) noexcept;

        /** @return Whether every group field equals @p expected. */
        bool allGroupsMatch(
            std::uint64_t MoEOverlayDeviceControllerGroupRecord::*field,
            std::uint64_t expected) const noexcept;

        MoEOverlayDeviceControllerSharedHeader *header_ = nullptr;
        std::span<MoEOverlayDeviceControllerGroupRecord> groups_;
        MoEOverlayDeviceControllerCommandHeader *command_ = nullptr;
    };
} // namespace llaminar2
