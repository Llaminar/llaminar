/**
 * @file StorageIOCoordinator.h
 * @brief Per-slot storage-I/O coordinator (design §9.6).
 *
 * Prefix restore/harvest overlap chooses disjoint slot/storage ownership, not
 * whole-domain serial restore. Every copy lane is pre-admitted with exact
 * non-null streams and its PMA bound; readiness is event-aware publication
 * (restorePublished requires the exact producer event frontier). Cancel/
 * reuse waits for the old generation's actual copy/snapshot borrowers;
 * TransferEngine owns copies, source retirement and readiness. This typed
 * per-slot state makes cross-slot serialization unrepresentable: each slot has
 * its own coordinator, and no transition can touch another slot.
 */

#pragma once

#include <cstdint>
#include <stdexcept>

namespace llaminar2
{
    /** @brief Per-slot storage-I/O lane phases (design §9.6). */
    enum class StorageIOLanePhase : std::uint8_t
    {
        Idle,
        RestoreAdmitted,
        RestoreInFlight,
        RestorePublished,       ///< Event-aware frontier proves readiness.
        SnapshotFrozen,         ///< Canonical harvest point; source frozen.
        SnapshotCopyInFlight,   ///< Copying into immutable snapshot staging.
        SnapshotBorrowed,       ///< Slower storage borrows the detached snapshot.
    };

    /** @brief One slot's storage-I/O state; cross-slot dependencies cannot be expressed. */
    struct SlotStorageIOCoordinator
    {
        StorageIOLanePhase phase = StorageIOLanePhase::Idle;
        std::uint32_t slot = 0;
        std::uint64_t generation = 0;

        void admitRestore()
        {
            transition(StorageIOLanePhase::Idle, StorageIOLanePhase::RestoreAdmitted);
        }
        void beginRestore()
        {
            transition(StorageIOLanePhase::RestoreAdmitted, StorageIOLanePhase::RestoreInFlight);
        }
        /** @brief Event-aware readiness: requires the exact producer event frontier. */
        void publishRestore(bool producer_frontier_published)
        {
            if (!producer_frontier_published)
                throw std::logic_error("Restore readiness requires the exact producer frontier");
            transition(StorageIOLanePhase::RestoreInFlight, StorageIOLanePhase::RestorePublished);
        }
        /** @brief Canonical harvest point: freeze the source until copy releases it. */
        void freezeSnapshot()
        {
            transition(StorageIOLanePhase::RestorePublished, StorageIOLanePhase::SnapshotFrozen);
        }
        void beginSnapshotCopy()
        {
            transition(StorageIOLanePhase::SnapshotFrozen, StorageIOLanePhase::SnapshotCopyInFlight);
        }
        /** @brief Copy released the source; slower storage borrows the detached snapshot. */
        void borrowSnapshot()
        {
            transition(StorageIOLanePhase::SnapshotCopyInFlight, StorageIOLanePhase::SnapshotBorrowed);
        }

    private:
        void transition(StorageIOLanePhase from, StorageIOLanePhase to)
        {
            if (phase != from)
                throw std::logic_error("Invalid storage-I/O lane transition");
            phase = to;
        }
    };
} // namespace llaminar2
