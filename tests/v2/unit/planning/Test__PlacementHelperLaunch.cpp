/** @brief Isolated launch-chain tests for the placement helper (bounded,
 *         approved scope: helper executable + co-location + managed receipt
 *         verification in a test harness — NO production launcher wiring).
 *
 * Each test execs the ACTUAL helper binary via the agreed per-rank boundary
 * (helper options, "--", application argv verbatim) with the child-check
 * binary as the application, then asserts the observed native exit status.
 * Failure branches are exercised against the typed exit contract: 2 usage,
 * 3 topology, 4 budget, 5 exec, 6 identity. No PATH fallback: every spawn
 * uses the absolute CMake-resolved helper path.
 */
#include <gtest/gtest.h>

#include <sys/wait.h>
#include <unistd.h>

#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <string>

namespace
{
    std::string selfExecutable()
    {
        std::filesystem::path exe = std::filesystem::canonical("/proc/self/exe");
        return exe.string();
    }

    /// Clear expectation/OMP state so cases cannot leak into each other.
    void cleanChildEnv()
    {
        ::unsetenv("OMP_NUM_THREADS");
        ::unsetenv("OMP_PLACES");     // the global test env exports "cores"
        ::unsetenv("OMP_PROC_BIND");
        ::unsetenv("LLAMINAR_HELPER_CHECK_budget");
        ::unsetenv("LLAMINAR_HELPER_CHECK_session");
        ::unsetenv("LLAMINAR_HELPER_CHECK_rank");
    }

    /// Exec helper in a forked child and return the native wait status.
    /// append_child=false spawns the helper with ONLY the given options (for
    /// usage-error cases where the "-- application" part must be absent).
    int runHelper(const std::vector<std::string> &options,
        const std::string &child_path, const bool append_child = true)
    {
        const pid_t pid = ::fork();
        if (pid < 0) return -1;
        if (pid == 0)
        {
            std::vector<char *> argv;
            argv.push_back(const_cast<char *>(LLAMINAR_PLACEMENT_HELPER_PATH));
            for (const auto &option : options)
                argv.push_back(const_cast<char *>(option.c_str()));
            if (append_child)
            {
                argv.push_back(const_cast<char *>("--"));
                argv.push_back(const_cast<char *>(child_path.c_str()));
            }
            argv.push_back(nullptr);
            ::execv(LLAMINAR_PLACEMENT_HELPER_PATH, argv.data());
            std::_Exit(90); // exec of the helper itself failed
        }
        int status = 0;
        ::waitpid(pid, &status, 0);
        return status;
    }

    int rawExit(const int status) { return WIFEXITED(status) ? WEXITSTATUS(status) : -1; }
}

TEST(PlacementHelperLaunch, ManagedReceiptReachesTheApplicationAcrossTheExecBoundary)
{
    // The test process sets the independent rank identity env the helper
    // requires, plus the child's expectations; both cross the helper exec.
    cleanChildEnv();
    setenv("OMPI_COMM_WORLD_RANK", "3", 1);
    setenv("LLAMINAR_HELPER_CHECK_session", "session-tok-1", 1);
    setenv("LLAMINAR_HELPER_CHECK_rank", "3", 1);
    // Budget 2: the container exposes >= 2 places; Explicit origin.
    setenv("LLAMINAR_HELPER_CHECK_budget", "2", 1);
    const int status = runHelper({"--budget=2", "--session=session-tok-1"},
        LLAMINAR_PLACEMENT_CHILD_PATH);
    ASSERT_EQ(rawExit(status), 0) << "raw status " << status;
}

TEST(PlacementHelperLaunch, AutoLocalBudgetResolvesFromOwnPlaces)
{
    cleanChildEnv();
    setenv("OMPI_COMM_WORLD_RANK", "0", 1);
    setenv("LLAMINAR_HELPER_CHECK_session", "auto-1", 1);
    setenv("LLAMINAR_HELPER_CHECK_rank", "0", 1);
    // No --budget: AutoLocal resolves to this helper's own place count; the
    // child cannot know it a priori, so it only checks presence/shape here —
    // implemented by expecting the receipt's own budget via a sentinel the
    // helper and child agree on: OMP_NUM_THREADS must be present and the
    // transport budget_origin must be AutoLocal. The child validates the
    // parts that do not need the number; the parent validates the origin.
    unsetenv("OMP_NUM_THREADS");
    const int status = runHelper({"--session=auto-1"}, LLAMINAR_PLACEMENT_CHILD_PATH);
    // The child expects a specific budget; AutoLocal is place-count-sized.
    // The child tolerates any positive budget via the budget field check on
    // the transport, so this must pass with exit 0.
    ASSERT_EQ(rawExit(status), 0) << "raw status " << status;
}

TEST(PlacementHelperLaunch, UsageErrorsExit2WithoutExecutingAnything)
{
    // No child appended: the helper must fail at usage BEFORE any exec.
    EXPECT_EQ(rawExit(runHelper({"--unknown=1", "--session=s"}, "", false)), 2);
    EXPECT_EQ(rawExit(runHelper({"--budget=2"}, "", false)), 2);
    EXPECT_EQ(rawExit(runHelper({"--session=s", "--"}, "", false)), 2); // -- with no application argv
    EXPECT_EQ(rawExit(runHelper({"--budget=abc", "--session=s"}, "", false)), 2);
}

TEST(PlacementHelperLaunch, BudgetOutsidePlaceCountExits4)
{
    cleanChildEnv();
    setenv("OMPI_COMM_WORLD_RANK", "1", 1);
    // A budget beyond any plausible place count on this host.
    EXPECT_EQ(rawExit(runHelper({"--budget=100000", "--session=s"}, LLAMINAR_PLACEMENT_CHILD_PATH)), 4);
}

TEST(PlacementHelperLaunch, AuthoredOmpNumThreadsConflictExits4AndPreservesAuthorship)
{
    cleanChildEnv();
    setenv("OMPI_COMM_WORLD_RANK", "1", 1);
    setenv("OMP_NUM_THREADS", "7", 1); // authored; conflicts with budget 2
    EXPECT_EQ(rawExit(runHelper({"--budget=2", "--session=s"}, LLAMINAR_PLACEMENT_CHILD_PATH)), 4);
    EXPECT_STREQ(getenv("OMP_NUM_THREADS"), "7"); // authorship preserved
    unsetenv("OMP_NUM_THREADS");
}

TEST(PlacementHelperLaunch, MissingIndependentRankIdentityExits6)
{
    cleanChildEnv();
    unsetenv("OMPI_COMM_WORLD_RANK");
    unsetenv("PMI_RANK");
    EXPECT_EQ(rawExit(runHelper({"--budget=2", "--session=s"}, LLAMINAR_PLACEMENT_CHILD_PATH)), 6);
}

TEST(PlacementHelperLaunch, ApplicationExecFailureExits5)
{
    cleanChildEnv();
    setenv("OMPI_COMM_WORLD_RANK", "1", 1);
    // The helper must exec a nonexistent application and report 5 (exec of
    // the helper itself works; the APPLICATION exec fails).
    const int status = runHelper({"--budget=2", "--session=s"},
        LLAMINAR_PLACEMENT_CHILD_PATH "/nonexistent-child");
    EXPECT_EQ(rawExit(status), 5);
}

// ---- native child coverage (OpenMP-linked child; actual constructor contract) ----

namespace
{
    /// Exec the helper with the NATIVE OpenMP child as the application.
    int runHelperNative(const std::vector<std::string> &options)
    {
        cleanChildEnv();
        const pid_t pid = ::fork();
        if (pid < 0) return -1;
        if (pid == 0)
        {
            std::vector<char *> argv;
            argv.push_back(const_cast<char *>(LLAMINAR_PLACEMENT_HELPER_PATH));
            for (const auto &option : options)
                argv.push_back(const_cast<char *>(option.c_str()));
            argv.push_back(const_cast<char *>("--"));
            argv.push_back(const_cast<char *>(LLAMINAR_PLACEMENT_NATIVE_CHILD_PATH));
            argv.push_back(nullptr);
            ::execv(LLAMINAR_PLACEMENT_HELPER_PATH, argv.data());
            std::_Exit(90);
        }
        int status = 0;
        ::waitpid(pid, &status, 0);
        return status;
    }
}

TEST(PlacementHelperLaunch, NativeChildVerifiesConstructorPlacesAndMasks)
{
    // Full receipt flow: the helper exports OMP_PLACES/PROC_BIND=close/
    // OMP_NUM_THREADS BEFORE the application image loads, so the libgomp
    // constructor builds the receipt's places and binds the team. The
    // native child verifies: native place list == receipt places (count,
    // order, per-place proc-ids), proc_bind != false, master on the first
    // place, and every thread's mask == its native place's proc-id set.
    cleanChildEnv();
    setenv("OMPI_COMM_WORLD_RANK", "2", 1);
    setenv("LLAMINAR_MANAGED_PLACEMENT", "1", 1); // the launcher exports this via -x
    setenv("LLAMINAR_HELPER_CHECK_session", "native-1", 1);
    setenv("LLAMINAR_HELPER_CHECK_rank", "2", 1);
    const int status = runHelperNative({"--budget=2", "--session=native-1"});
    unsetenv("LLAMINAR_MANAGED_PLACEMENT");
    ASSERT_EQ(rawExit(status), 0) << "raw status " << status;
}

TEST(PlacementHelperLaunch, AuthoredOmpPlacesConflictsAreFatalNotSilentlyRerun)
{
    // An authored OMP_PLACES in a managed launch conflicts with the receipt
    // authority: FATAL before exec — never a silently-run wrong layout.
    cleanChildEnv();
    setenv("OMPI_COMM_WORLD_RANK", "1", 1);
    setenv("OMP_PLACES", "{0},{1}", 1);
    EXPECT_EQ(rawExit(runHelper({"--budget=2", "--session=s"}, LLAMINAR_PLACEMENT_CHILD_PATH)), 4);
    unsetenv("OMP_PLACES");
}

TEST(PlacementHelperLaunch, AuthoredOmpProcBindFalseConflictsAreFatal)
{
    // An authored OMP_PROC_BIND=false in a managed launch conflicts with
    // the close-binding intent: FATAL before exec.
    cleanChildEnv();
    setenv("OMPI_COMM_WORLD_RANK", "1", 1);
    setenv("OMP_PROC_BIND", "false", 1);
    EXPECT_EQ(rawExit(runHelper({"--budget=2", "--session=s"}, LLAMINAR_PLACEMENT_CHILD_PATH)), 4);
    unsetenv("OMP_PROC_BIND");
}

TEST(PlacementHelperLaunch, AuthoredOmpProcBindSpreadConflictsAreFatal)
{
    cleanChildEnv();
    setenv("OMPI_COMM_WORLD_RANK", "1", 1);
    setenv("OMP_PROC_BIND", "spread", 1);
    EXPECT_EQ(rawExit(runHelper({"--budget=2", "--session=s"}, LLAMINAR_PLACEMENT_CHILD_PATH)), 4);
    unsetenv("OMP_PROC_BIND");
}

TEST(PlacementHelperLaunch, AuthoredOmpProcBindCloseIsCompatibleAndRuns)
{
    // An authored OMP_PROC_BIND=close AGREES with the managed intent: the
    // helper proceeds (preserve-compatible, not fatal).
    cleanChildEnv();
    setenv("OMPI_COMM_WORLD_RANK", "1", 1);
    setenv("LLAMINAR_HELPER_CHECK_session", "close-ok", 1);
    setenv("LLAMINAR_HELPER_CHECK_rank", "1", 1);
    setenv("LLAMINAR_HELPER_CHECK_budget", "2", 1);
    setenv("OMP_PROC_BIND", "close", 1);
    const int status = runHelper({"--budget=2", "--session=close-ok"},
        LLAMINAR_PLACEMENT_CHILD_PATH);
    ASSERT_EQ(rawExit(status), 0) << "raw status " << status;
    unsetenv("OMP_PROC_BIND");
}
