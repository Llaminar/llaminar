/**
 * @file Test__ParityCellLifecycle.cpp
 * @brief Real-MPI integration proof for parity cell control/teardown isolation.
 *
 * These tests intentionally load no model and touch no accelerator. They prove
 * the communicator protocol used by every production parity fixture at one and
 * multiple ranks, while the source-policy unit proves that ParityTestBase is
 * the sole caller of the protocol.
 */

#include "integration/parity/ParityCellLifecycle.h"

#include <gtest/gtest.h>
#include <mpi.h>

namespace llaminar2::test::parity
{
    /**
     * @brief Prove both lanes exist, are isolated from WORLD, and retire once.
     */
    TEST(Test__ParityCellLifecycle, OwnsIsolatedChannelsAndRetires)
    {
        int world_rank = -1;
        int world_size = 0;
        ASSERT_EQ(MPI_Comm_rank(MPI_COMM_WORLD, &world_rank), MPI_SUCCESS);
        ASSERT_EQ(MPI_Comm_size(MPI_COMM_WORLD, &world_size), MPI_SUCCESS);

        ParityCellLifecycle lifecycle;
        EXPECT_EQ(
            lifecycle.state(),
            ParityCellLifecycleState::NotEntered);
        EXPECT_FALSE(lifecycle.active());

        lifecycle.enter(MPI_COMM_WORLD);
        ASSERT_TRUE(lifecycle.active());
        EXPECT_EQ(lifecycle.state(), ParityCellLifecycleState::Active);

        const auto &control = lifecycle.controlContext();
        ASSERT_NE(control, nullptr);
        EXPECT_EQ(control->rank(), world_rank);
        EXPECT_EQ(control->world_size(), world_size);

        int relationship = MPI_UNEQUAL;
        ASSERT_EQ(
            MPI_Comm_compare(
                MPI_COMM_WORLD,
                control->communicator(),
                &relationship),
            MPI_SUCCESS);
        EXPECT_EQ(relationship, MPI_CONGRUENT)
            << "The control lane must preserve rank order without aliasing WORLD";

        // Exercise both protocol lanes in their production order.
        control->barrier();
        lifecycle.teardownBarrier();
        lifecycle.teardownBarrier();
        lifecycle.retire();

        EXPECT_EQ(lifecycle.state(), ParityCellLifecycleState::Retired);
        EXPECT_FALSE(lifecycle.active());
        EXPECT_THROW(
            static_cast<void>(lifecycle.controlContext()),
            std::logic_error);
        EXPECT_THROW(lifecycle.teardownBarrier(), std::logic_error);
        EXPECT_THROW(lifecycle.retire(), std::logic_error);
    }

    /** @brief Reject null sources, re-entry, and pre-entry operations. */
    TEST(Test__ParityCellLifecycle, RejectsInvalidTransitions)
    {
        ParityCellLifecycle lifecycle;
        EXPECT_THROW(
            static_cast<void>(lifecycle.controlContext()),
            std::logic_error);
        EXPECT_THROW(lifecycle.teardownBarrier(), std::logic_error);
        EXPECT_THROW(lifecycle.retire(), std::logic_error);
        EXPECT_THROW(lifecycle.enter(MPI_COMM_NULL), std::invalid_argument);
        EXPECT_EQ(
            lifecycle.state(),
            ParityCellLifecycleState::NotEntered);

        lifecycle.enter(MPI_COMM_WORLD);
        EXPECT_THROW(lifecycle.enter(MPI_COMM_WORLD), std::logic_error);
        lifecycle.teardownBarrier();
        lifecycle.retire();
    }
} // namespace llaminar2::test::parity

/** @brief Initialize/finalize MPI around the real communicator integration. */
int main(int argc, char **argv)
{
    int provided = MPI_THREAD_SINGLE;
    if (MPI_Init_thread(
            &argc,
            &argv,
            MPI_THREAD_FUNNELED,
            &provided) != MPI_SUCCESS ||
        provided < MPI_THREAD_FUNNELED)
    {
        return 2;
    }

    ::testing::InitGoogleTest(&argc, argv);
    const int result = RUN_ALL_TESTS();
    if (MPI_Finalize() != MPI_SUCCESS)
        return 3;
    return result;
}
