/**
 * @file Test__ServingPublicationDrain.cpp
 * @brief §10.1 transport-completion edge: exactly-once delivery release.
 *
 * The retained-delivery count (Snapshot.delivering) must be released exactly
 * once, at the typed transport-completion edge (publicationFinished), and the
 * abandoned path (client drops before completion) must still release capacity.
 * Device-free deterministic state-machine tests.
 */

#include <gtest/gtest.h>

#include "app/modes/SerializedInferenceExecutor.h"

namespace llaminar2::test
{
    TEST(ServingPublicationDrain, CompletionReleasesDeliveryExactlyOnce)
    {
        SerializedInferenceExecutor executor(8, 4);
        auto reservation = executor.tryReserve();
        ASSERT_TRUE(reservation);
        reservation->run([] {});
        const auto after_run = executor.snapshot();
        ASSERT_EQ(after_run.delivering, 1u);
        reservation->publicationInFlight();
        reservation->publicationDetached();
        reservation->publicationFinished();
        EXPECT_EQ(executor.snapshot().delivering, 0u);
        // Exactly-once: a second completion is refused, and no second release
        // happens (the destructor's abandoned path is disarmed).
        EXPECT_THROW(reservation->publicationFinished(), std::logic_error);
        EXPECT_EQ(executor.snapshot().delivering, 0u);
        reservation.reset();
        EXPECT_EQ(executor.snapshot().delivering, 0u);
        EXPECT_EQ(executor.snapshot().admitted, 0u);
    }

    TEST(ServingPublicationDrain, AbandonedDeliveryReleasesCapacityOnDrop)
    {
        SerializedInferenceExecutor executor(8, 4);
        auto reservation = executor.tryReserve();
        ASSERT_TRUE(reservation);
        reservation->run([] {});
        ASSERT_EQ(executor.snapshot().delivering, 1u);
        // Client leaves before transport completion: capacity is released on
        // drop, no exception, no double release.
        reservation.reset();
        EXPECT_EQ(executor.snapshot().delivering, 0u);
        EXPECT_EQ(executor.snapshot().admitted, 0u);
    }

    TEST(ServingPublicationDrain, MachineRejectsOutOfOrder)
    {
        SerializedInferenceExecutor executor(8, 4);
        auto reservation = executor.tryReserve();
        ASSERT_TRUE(reservation);
        reservation->run([] {});
        EXPECT_THROW(reservation->publicationFinished(), std::logic_error); // reserved
        reservation->publicationInFlight();
        EXPECT_THROW(reservation->publicationFinished(), std::logic_error); // in flight
        reservation->publicationDetached();
        EXPECT_NO_THROW(reservation->publicationFinished());
    }

    TEST(ServingPublicationDrain, PublicationBeforeRunIsRefused)
    {
        SerializedInferenceExecutor executor(8, 4);
        auto reservation = executor.tryReserve();
        ASSERT_TRUE(reservation);
        // Counterexample regression: publication machine must not run before run().
        EXPECT_THROW(reservation->publicationInFlight(), std::logic_error);
        EXPECT_THROW(reservation->publicationDetached(), std::logic_error);
        EXPECT_THROW(reservation->publicationFinished(), std::logic_error);
        EXPECT_EQ(executor.snapshot().delivering, 0u); // never underflowed
        reservation.reset();
        EXPECT_EQ(executor.snapshot().delivering, 0u);
    }

    TEST(ServingPublicationDrain, TransportFailureTakesTheAbandonedPath)
    {
        // §10.1 terminal ownership: when the transport fails (chunk write rejected)
        // the server must NOT emit a fabricated success close. The reservation
        // takes the abandoned path: delivering is released by destruction, the
        // lease is closed, and no Drained record exists.
        auto executor = std::make_shared<SerializedInferenceExecutor>();
        auto reservation = executor->tryReserve();
        ASSERT_NE(reservation, nullptr);
        reservation->run([&] { /* production finished */ });
        // Simulate the transport outcome: NO publication calls (write failure).
        EXPECT_EQ(executor->snapshot().delivering, 1u);
        reservation.reset(); // abandoned: destructor releases delivering honestly
        EXPECT_EQ(executor->snapshot().delivering, 0u);
        EXPECT_EQ(executor->snapshot().admitted, 0u);
    }
} // namespace llaminar2::test
