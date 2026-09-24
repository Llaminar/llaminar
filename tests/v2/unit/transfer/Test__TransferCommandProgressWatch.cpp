/**
 * @file Test__TransferCommandProgressWatch.cpp
 * @brief Device-free proofs of command-local pending-transfer diagnostics.
 *
 * Injected monotonic timestamps cover permanently busy queues, generation reuse,
 * independently stalled slots, throttling, and counter boundaries without sleep.
 */
#include "transfer/TransferCommandProgressWatch.h"
#include <gtest/gtest.h>
#include <limits>

namespace llaminar2::test
{
    /** @brief Many fast commands never inherit a busy queue's accumulated age. */
    TEST(TransferCommandProgressWatch, BusyQueueDoesNotAgeNewCommands)
    {
        TransferCommandProgressWatch watch;
        for (std::uint64_t generation = 0; generation < 1000u; ++generation)
        {
            const auto start = generation * 100'000'000u;
            watch.published(start);
            EXPECT_FALSE(watch.pending(start + 99'999'999u));
        }
    }

    /** @brief A genuinely stalled command warns regardless of other progress. */
    TEST(TransferCommandProgressWatch, StalledSlotHasIndependentThrottledAge)
    {
        TransferCommandProgressWatch stalled;
        TransferCommandProgressWatch active;
        stalled.published(0u);
        for (std::uint64_t second = 1; second <= 10; ++second)
        {
            const auto now = second * 1'000'000'000u;
            active.published(now - 1u);
            EXPECT_FALSE(active.pending(now));
            const auto diagnostic = stalled.pending(now);
            if (second % 5 == 0)
                EXPECT_EQ(diagnostic, now);
            else
                EXPECT_FALSE(diagnostic);
        }
        stalled.published(11'000'000'000u);
        EXPECT_FALSE(stalled.pending(11'000'000'001u));
    }

    /** @brief Clock regression and near-overflow inputs cannot fabricate age. */
    TEST(TransferCommandProgressWatch, MonotonicBoundariesDoNotOverflow)
    {
        TransferCommandProgressWatch watch;
        const auto maximum = std::numeric_limits<std::uint64_t>::max();
        watch.published(maximum - 5'000'000'000ull);
        EXPECT_FALSE(watch.pending(0u));
        EXPECT_EQ(watch.pending(maximum), 5'000'000'000ull);
        EXPECT_FALSE(watch.pending(maximum));
    }

    /** @brief Host service delay is distinct from a recently incomplete event. */
    TEST(TransferCommandProgressWatch, NativeObservationDoesNotInventGpuProgress)
    {
        TransferCommandProgressWatch watch;
        watch.published(100u);
        EXPECT_EQ(watch.incompleteEventQueries(), 0u);
        EXPECT_FALSE(watch.incompleteEventQueryAge(5'000'000'100ull));
        watch.observedIncompleteEvent(150u);
        EXPECT_EQ(watch.incompleteEventQueries(), 1u);
        EXPECT_EQ(watch.incompleteEventQueryAge(160u), 10u);
        // An overdue command without a recent worker poll is not proof that
        // the GPU has remained busy throughout this interval.
        EXPECT_EQ(watch.incompleteEventQueryAge(5'000'000'150ull),
                  5'000'000'000ull);
        EXPECT_EQ(watch.incompleteEventQueries(), 1u);
        watch.observedIncompleteEvent(5'000'000'150ull);
        EXPECT_EQ(watch.incompleteEventQueries(), 2u);
        EXPECT_EQ(watch.incompleteEventQueryAge(5'000'000'151ull), 1u);
    }

    /** @brief Reuse and clock regression cannot leak another generation's age. */
    TEST(TransferCommandProgressWatch, NativeObservationsResetWithPublication)
    {
        TransferCommandProgressWatch watch;
        watch.published(100u);
        watch.observedIncompleteEvent(120u);
        watch.observedIncompleteEvent(119u);
        EXPECT_EQ(watch.incompleteEventQueries(), 1u);
        EXPECT_FALSE(watch.incompleteEventQueryAge(119u));
        watch.published(200u);
        EXPECT_EQ(watch.incompleteEventQueries(), 0u);
        EXPECT_FALSE(watch.incompleteEventQueryAge(205u));
        watch.observedIncompleteEvent(199u);
        EXPECT_EQ(watch.incompleteEventQueries(), 0u);
        watch.observedIncompleteEvent(201u);
        EXPECT_EQ(watch.incompleteEventQueryAge(202u), 1u);
    }
} // namespace llaminar2::test
