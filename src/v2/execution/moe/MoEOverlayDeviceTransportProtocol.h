/**
 * @file MoEOverlayDeviceTransportProtocol.h
 * @brief Typed host transport follower for device-authored ExpertOverlay waves.
 *
 * An all-GPU overlay keeps placement policy and durable epoch publication on
 * the authority GPU. A host worker is nevertheless allowed to progress a
 * node-local, cross-vendor, or network byte transport. This file makes that
 * exception narrow: the worker can acquire one immutable command batch and
 * publish only physical prepared/published/retired/restored completion edges.
 * It cannot access histogram pages, author commands, mutate the controller, or
 * construct a host owner-map mirror.
 */

#pragma once

#include "MoEOverlayNodeLocalDeviceControllerFabric.h"

#include <cstdint>
#include <string>
#include <vector>

namespace llaminar2
{
    /** Immutable device-authored command bytes retained by one physical wave. */
    struct MoEOverlayDeviceTransportCommandBatch
    {
        MoEOverlayDeviceControllerCommandHeader header;
        std::vector<MoEOverlayDeviceMovementCommand> entries;
        std::uint32_t participant_count = 0u;
        std::uint32_t num_layers = 0u;
        std::uint32_t num_experts = 0u;

        /** @return Whether header, entries, digest, and geometry are coherent. */
        [[nodiscard]] bool valid() const noexcept;

        /** @return Whether this command requires physical weight transport. */
        [[nodiscard]] bool movesWeights() const noexcept
        {
            return header.packed_weight_bytes != 0u;
        }
    };

    /** Non-blocking immutable-command acquisition result. */
    enum class MoEOverlayDeviceTransportAcquireStatus
    {
        Waiting, ///< No command newer than the caller's observed transaction.
        Ready,   ///< A complete authenticated immutable batch was acquired.
        Failed,  ///< Shared identity or command bytes were invalid.
    };

    /** Owned result of polling one group-local command transport lane. */
    struct MoEOverlayDeviceTransportAcquireResult
    {
        MoEOverlayDeviceTransportAcquireStatus status =
            MoEOverlayDeviceTransportAcquireStatus::Waiting;
        MoEOverlayDeviceTransportCommandBatch batch;
        std::string error;
    };

    /**
     * @brief Authenticate device commands and publish physical completion only.
     *
     * All methods are non-blocking. A background service polls command and
     * backend/network events, then calls the matching publication method. The
     * group-root GPU acquires this record and remains the only writer of its
     * controller acknowledgement lane.
     */
    class MoEOverlayDeviceTransportProtocol final
    {
    public:
        /**
         * @brief Bind one exact local group-root transport lane.
         * @throws std::invalid_argument for an incomplete or foreign binding.
         */
        explicit MoEOverlayDeviceTransportProtocol(
            MoEOverlayDeviceControllerTransportBinding binding);

        /**
         * @brief Poll for and authenticate one newer immutable command batch.
         * @param after_transaction Last transaction already consumed locally.
         * @return Waiting, an owned exact batch, or a fatal diagnostic.
         */
        [[nodiscard]] MoEOverlayDeviceTransportAcquireResult tryAcquire(
            std::uint64_t after_transaction) noexcept;

        /**
         * @brief Observe the immutable ticket for a new snapshot transaction.
         * @param after_transaction Last transaction completed by this worker.
         * @param transaction Receives the new transaction when collection is open.
         * @param kind Receives the authority-selected transaction objective.
         * @param phase Receives the phase-pure plane, or Invalid for restore.
         * @return True only for an exact newer durable transaction.
         *
         * This is a scheduler ticket, not a host policy mirror. The host learns
         * only the monotonic transaction identity needed to submit retained
         * participant snapshot graphs.
         */
        [[nodiscard]] bool snapshotTransactionAfter(
            std::uint64_t after_transaction,
            std::uint64_t *transaction,
            MoEOverlayDeviceControllerTransactionKind *kind,
            MoEOverlayDeviceDemandPhase *phase) const noexcept;

        /** @return Whether every local participant published @p transaction. */
        [[nodiscard]] bool localSnapshotsReady(
            std::uint64_t transaction) const noexcept;

        /** @return Whether every group root published @p transaction. */
        [[nodiscard]] bool allGroupsSnapshotted(
            std::uint64_t transaction) const noexcept;

        /** Publish completion of all transfer and preparation operations. */
        [[nodiscard]] bool publishPrepared(
            const MoEOverlayDeviceTransportCommandBatch &batch,
            std::string *error = nullptr) noexcept;

        /**
         * @return Whether physical preparation and every local candidate are ready.
         *
         * A group-root acknowledgement graph is submitted only after this
         * monotonic predicate is true, keeping its device action bounded.
         */
        [[nodiscard]] bool preparationReady(
            const MoEOverlayDeviceTransportCommandBatch &batch) const noexcept;

        /** @return Whether every group has acknowledged candidate preparation. */
        [[nodiscard]] bool allGroupsPrepared(
            const MoEOverlayDeviceTransportCommandBatch &batch) const noexcept;

        /** @return Whether the authority GPU has opened local bank commit. */
        [[nodiscard]] bool commitRequested(
            const MoEOverlayDeviceTransportCommandBatch &batch) const noexcept;

        /**
         * @return Whether commit is open and every local device published RCU.
         *
         * A worker polls this before performing its cheap transfer-side commit
         * and calling @ref publishPublished. It observes participant lifecycle
         * words only; placement and inference state remain inaccessible.
         */
        [[nodiscard]] bool publicationReady(
            const MoEOverlayDeviceTransportCommandBatch &batch) const noexcept;

        /**
         * @return Whether the sole authority rejected this exact transaction.
         *
         * A physical worker uses this terminal observation to stop polling
         * backend events after a peer participant fails. It does not expose a
         * policy, owner map, or mutable controller field to the host.
         */
        [[nodiscard]] bool authorityRejected(
            const MoEOverlayDeviceTransportCommandBatch &batch) const noexcept;

        /**
         * Publish durable physical-candidate readiness before device RCU apply.
         *
         * Destination bytes, descriptors, and ownership must already be
         * complete. The record may arrive while the bounded publication graph
         * is applying local candidates, but must precede global admission. It
         * does not itself select a runtime bank or admit inference.
         */
        [[nodiscard]] bool publishPublished(
            const MoEOverlayDeviceTransportCommandBatch &batch,
            std::string *error = nullptr) noexcept;

        /** @return Whether local runtime and physical publication are complete. */
        [[nodiscard]] bool groupPublicationReady(
            const MoEOverlayDeviceTransportCommandBatch &batch) const noexcept;

        /** @return Whether every group acknowledged runtime publication. */
        [[nodiscard]] bool allGroupsPublished(
            const MoEOverlayDeviceTransportCommandBatch &batch) const noexcept;

        /** @return Whether device-authored admission opened durable retirement. */
        [[nodiscard]] bool retirementOpen(
            const MoEOverlayDeviceTransportCommandBatch &batch) const noexcept;

        /**
         * @return Whether every topology device published its old-reader receipt.
         *
         * This is the authenticated scheduler ticket for the bounded local-bank
         * retirement graph. The host may observe it and submit that retained
         * graph, but cannot write the participant records or choose an epoch.
         */
        [[nodiscard]] bool runtimeReadersReady(
            const MoEOverlayDeviceTransportCommandBatch &batch) const noexcept;

        /** @return Whether the authority GPU opened old-epoch retirement. */
        [[nodiscard]] bool retirementRequested(
            const MoEOverlayDeviceTransportCommandBatch &batch) const noexcept;

        /** Publish reclamation of the exact prior durable epoch. */
        [[nodiscard]] bool publishRetired(
            const MoEOverlayDeviceTransportCommandBatch &batch,
            std::string *error = nullptr) noexcept;

        /** @return Whether local device and physical retirement are complete. */
        [[nodiscard]] bool groupRetirementReady(
            const MoEOverlayDeviceTransportCommandBatch &batch) const noexcept;

        /** @return Whether every group acknowledged the prior epoch retirement. */
        [[nodiscard]] bool allGroupsRetired(
            const MoEOverlayDeviceTransportCommandBatch &batch) const noexcept;

        /** @return Whether the sole device authority completed this transaction. */
        [[nodiscard]] bool transactionComplete(
            const MoEOverlayDeviceTransportCommandBatch &batch) const noexcept;

        /** @return Whether the authority GPU opened transient LLEP restoration. */
        [[nodiscard]] bool restorationRequested(
            const MoEOverlayDeviceTransportCommandBatch &batch) const noexcept;

        /** Publish reclamation of request-scoped LLEP physical arrivals. */
        [[nodiscard]] bool publishRestored(
            const MoEOverlayDeviceTransportCommandBatch &batch,
            std::string *error = nullptr) noexcept;

        /** Latch a fatal physical error so every device wait exits. */
        void fail() noexcept;

        /**
         * @brief Describe the current mapped lifecycle without policy payloads.
         *
         * The snapshot intentionally contains only controller, group,
         * participant, and physical-transport monotonic words. It is suitable
         * for a fatal timeout diagnostic but cannot expose histograms,
         * placement maps, or mutable device-owned policy state to the host.
         *
         * @return Compact group-scoped lifecycle description.
         */
        [[nodiscard]] std::string describeLifecycle() const;

        /** @return Exact immutable group id served by this protocol. */
        [[nodiscard]] int groupId() const noexcept
        {
            return binding_.group_id;
        }

    private:
        /** Validate that @p batch is the exact acquired record identity. */
        [[nodiscard]] bool requireAcquired(
            const MoEOverlayDeviceTransportCommandBatch &batch,
            std::string *error) noexcept;

        MoEOverlayDeviceControllerTransportBinding binding_;
    };
} // namespace llaminar2
