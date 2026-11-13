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
#include "utils/MPISessionOwner.h"
#include <type_traits>
#include <cstdlib>
#include <optional>
#include <string>
#include <stdexcept>

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

namespace
{
    /** @brief Restore original launch metadata without creating any MPI process. */
    class ScopedExternalMPIEnvironment final
    {
    public:
        explicit ScopedExternalMPIEnvironment(const char *name) : name_(name)
        {
            if (const char *old = std::getenv(name)) original_ = old;
            if (::setenv(name,"2",1) != 0) throw std::runtime_error("Cannot prepare external MPI fixture");
        }
        ~ScopedExternalMPIEnvironment()
        { if (original_) ::setenv(name_.c_str(),original_->c_str(),1); else ::unsetenv(name_.c_str()); }
    private:
        std::string name_;
        std::optional<std::string> original_;
    };


}

TEST(MPIProcessSession, ExternalLaunchMetadataNeverAcquiresOrDeletesParentBacking)
{
    const auto original = std::getenv("TMPDIR") ? std::optional<std::string>(std::getenv("TMPDIR")) : std::nullopt;
    for (const char *name : {"OMPI_COMM_WORLD_SIZE", "PMI_SIZE", "PMIX_NAMESPACE", "MPI_LOCALRANKID"}) {
        ScopedExternalMPIEnvironment external(name);
        EXPECT_NO_THROW(MPISessionOwner::requireProcessOwner());
        const auto retained = std::getenv("TMPDIR") ? std::optional<std::string>(std::getenv("TMPDIR")) : std::nullopt;
        EXPECT_EQ(retained,original);
    }
}
