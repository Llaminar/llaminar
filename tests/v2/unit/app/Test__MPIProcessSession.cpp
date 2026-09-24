/**
 * @file Test__MPIProcessSession.cpp
 * @brief Pure ownership checks for process sessions and typed startup results.
 *
 * These fixtures never initialize MPI. Actual process finalization and weak
 * context-owner retirement are covered by the standalone Integration proof.
 */
#include "app/RuntimeInitPhase.h"
#include "app/commands/CommandMPI.h"
#include "execution/runner/IOrchestrationRunnerFactory.h"
#include <gtest/gtest.h>
#include <type_traits>

using namespace llaminar2;

TEST(MPIProcessSession, EmptyAndMovedScopesDoNotAcquireMPI)
{
    MPIProcessSession empty;
    EXPECT_FALSE(empty.ownsMPI());
    MPIProcessSession moved(std::move(empty));
    EXPECT_FALSE(empty.ownsMPI());
    EXPECT_FALSE(moved.ownsMPI());
}

TEST(MPIProcessSession, OwnershipCannotBeCopiedOrReplacedUnderDependents)
{
    static_assert(!std::is_copy_constructible_v<MPIProcessSession>);
    static_assert(!std::is_move_assignable_v<MPIProcessSession>);
    static_assert(std::is_nothrow_move_constructible_v<AppContext>);
    static_assert(!std::is_move_assignable_v<AppContext>);
    static_assert(!std::is_move_assignable_v<CommandMPISession>);
    AppContext context;
    AppContext moved(std::move(context));
    EXPECT_FALSE(context.mpi_session.ownsMPI());
    EXPECT_FALSE(moved.mpi_session.ownsMPI());
}

TEST(MPIProcessSession, TerminalStartupResultsAreIndependentOfRequestedFlags)
{
    for (const auto terminal : {RuntimeInitExit::Completed, RuntimeInitExit::Failed})
    {
        RuntimeInitResult result(terminal);
        EXPECT_FALSE(std::holds_alternative<AppContext>(result));
        EXPECT_EQ(std::get<RuntimeInitExit>(result), terminal);
    }
    EXPECT_EQ(static_cast<int>(RuntimeInitExit::Completed), 0);
    EXPECT_EQ(static_cast<int>(RuntimeInitExit::Failed), 1);
}

TEST(MPIProcessSession, SelectedFactoryRejectsMissingMembershipWithoutEnteringMPI)
{
    EXPECT_THROW(createOrchestrationRunnerFactory(std::shared_ptr<IMPIContext>{}),
                 std::invalid_argument);
}
