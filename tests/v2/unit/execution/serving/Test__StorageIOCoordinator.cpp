/**
 * @file Test__StorageIOCoordinator.cpp
 * @brief §9.6 per-slot storage-I/O overlap contract.
 *
 * Restore/harvest lanes are per-slot (cross-slot serialization
 * unrepresentable); readiness is event-aware (publishRestore requires the
 * exact producer frontier); harvest freezes the source until the copy
 * releases it. Device-free.
 */

#include <gtest/gtest.h>

#include "execution/serving/StorageIOCoordinator.h"

namespace llaminar2::test
{
    TEST(StorageIOCoordinator, LegalRestoreAndHarvestChain)
    {
        SlotStorageIOCoordinator io;
        io.slot = 2;
        io.admitRestore();
        io.beginRestore();
        EXPECT_THROW(io.publishRestore(false), std::logic_error); // frontier not published
        io.publishRestore(true);
        io.freezeSnapshot();
        io.beginSnapshotCopy();
        io.borrowSnapshot();
        EXPECT_EQ(io.phase, StorageIOLanePhase::SnapshotBorrowed);
    }

    TEST(StorageIOCoordinator, PerSlotIndependenceIsStructural)
    {
        SlotStorageIOCoordinator a, c;
        a.slot = 1;
        c.slot = 3;
        a.admitRestore();
        a.beginRestore();
        a.publishRestore(true);
        // C's coordinator is untouched by A's progress (no cross-slot edge exists).
        EXPECT_EQ(c.phase, StorageIOLanePhase::Idle);
        c.admitRestore();
        c.beginRestore();
        EXPECT_EQ(c.phase, StorageIOLanePhase::RestoreInFlight);
    }

    TEST(StorageIOCoordinator, InvalidTransitionsRefused)
    {
        SlotStorageIOCoordinator io;
        EXPECT_THROW(io.beginRestore(), std::logic_error);  // not admitted
        io.admitRestore();
        EXPECT_THROW(io.freezeSnapshot(), std::logic_error); // not published
    }
} // namespace llaminar2::test
