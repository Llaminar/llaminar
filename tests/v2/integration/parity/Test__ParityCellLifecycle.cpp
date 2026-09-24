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
    /** @brief Prove a passing cell follows the complete typed retirement path. */
    TEST(Test__ParityCellLifecycle, ConvergesPassingCellAndRetires)
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

        // Exercise both protocol lanes in their only legal production order.
        control->barrier();
        lifecycle.beginTeardown();
        EXPECT_EQ(
            lifecycle.state(),
            ParityCellLifecycleState::TeardownEntered);
        const auto consensus = lifecycle.convergeOutcome(
            ParityCellLocalOutcome::Passed);
        EXPECT_TRUE(consensus.allRanksPassed());
        EXPECT_EQ(consensus.originating_failed_ranks, 0);
        EXPECT_EQ(consensus.participant_count, world_size);
        EXPECT_EQ(
            lifecycle.state(),
            ParityCellLifecycleState::OutcomeConverged);
        lifecycle.finishTeardown();

        EXPECT_EQ(lifecycle.state(), ParityCellLifecycleState::Retired);
        EXPECT_FALSE(lifecycle.active());
        EXPECT_THROW(
            static_cast<void>(lifecycle.controlContext()),
            std::logic_error);
        EXPECT_THROW(lifecycle.beginTeardown(), std::logic_error);
        EXPECT_THROW(
            static_cast<void>(lifecycle.convergeOutcome(
                ParityCellLocalOutcome::Passed)),
            std::logic_error);
        EXPECT_THROW(lifecycle.finishTeardown(), std::logic_error);
    }

    /**
     * @brief Make one MPI rank fail and prove every rank receives that result.
     *
     * This is the model-free regression for the campaign deadlock where rank
     * zero failed a numerical expectation, its peer passed, and GoogleTest
     * fail-fast allowed only the peer to enter the next generated model cell.
     */
    TEST(Test__ParityCellLifecycle, ConvergesOneRankFailureAcrossEveryRank)
    {
        int world_rank = -1;
        int world_size = 0;
        ASSERT_EQ(MPI_Comm_rank(MPI_COMM_WORLD, &world_rank), MPI_SUCCESS);
        ASSERT_EQ(MPI_Comm_size(MPI_COMM_WORLD, &world_size), MPI_SUCCESS);

        ParityCellLifecycle lifecycle;
        lifecycle.enter(MPI_COMM_WORLD);
        lifecycle.beginTeardown();

        const auto consensus = lifecycle.convergeOutcome(
            world_rank == 0
                ? ParityCellLocalOutcome::Failed
                : ParityCellLocalOutcome::Passed);
        EXPECT_FALSE(consensus.allRanksPassed());
        EXPECT_EQ(
            consensus.outcome,
            ParityCellAggregateOutcome::AtLeastOneRankFailed);
        EXPECT_EQ(consensus.originating_failed_ranks, 1);
        EXPECT_EQ(consensus.participant_count, world_size);

        lifecycle.finishTeardown();
        EXPECT_EQ(lifecycle.state(), ParityCellLifecycleState::Retired);
    }

    /** @brief Reject null sources, re-entry, and pre-entry operations. */
    TEST(Test__ParityCellLifecycle, RejectsInvalidTransitions)
    {
        ParityCellLifecycle lifecycle;
        EXPECT_THROW(
            static_cast<void>(lifecycle.controlContext()),
            std::logic_error);
        EXPECT_THROW(lifecycle.beginTeardown(), std::logic_error);
        EXPECT_THROW(
            static_cast<void>(lifecycle.convergeOutcome(
                ParityCellLocalOutcome::Passed)),
            std::logic_error);
        EXPECT_THROW(lifecycle.finishTeardown(), std::logic_error);
        EXPECT_THROW(lifecycle.enter(MPI_COMM_NULL), std::invalid_argument);
        EXPECT_EQ(
            lifecycle.state(),
            ParityCellLifecycleState::NotEntered);

        lifecycle.enter(MPI_COMM_WORLD);
        EXPECT_THROW(lifecycle.enter(MPI_COMM_WORLD), std::logic_error);
        EXPECT_THROW(
            static_cast<void>(lifecycle.convergeOutcome(
                ParityCellLocalOutcome::Passed)),
            std::logic_error);
        EXPECT_THROW(lifecycle.finishTeardown(), std::logic_error);

        lifecycle.beginTeardown();
        EXPECT_THROW(lifecycle.beginTeardown(), std::logic_error);
        EXPECT_THROW(
            static_cast<void>(lifecycle.controlContext()),
            std::logic_error);
        EXPECT_THROW(lifecycle.finishTeardown(), std::logic_error);
        EXPECT_THROW(
            static_cast<void>(lifecycle.convergeOutcome(
                static_cast<ParityCellLocalOutcome>(255))),
            std::invalid_argument);

        static_cast<void>(lifecycle.convergeOutcome(
            ParityCellLocalOutcome::Passed));
        EXPECT_THROW(
            static_cast<void>(lifecycle.convergeOutcome(
                ParityCellLocalOutcome::Passed)),
            std::logic_error);
        EXPECT_THROW(lifecycle.beginTeardown(), std::logic_error);
        lifecycle.finishTeardown();
        EXPECT_THROW(lifecycle.finishTeardown(), std::logic_error);
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
