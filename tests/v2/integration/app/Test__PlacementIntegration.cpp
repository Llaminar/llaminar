/** @brief Isolated OpenMP placement diagnostics and MPI session retirement.
 *
 * Explicitly composes the diagnostic helper with real MPI ranks; the helper
 * does not participate in product placement. The child checks native OpenMP
 * policy and worker occupancy. Cancellation tests use the production session
 * owner to terminate and join the diagnostic process tree and verify retirement.
 */
#include <gtest/gtest.h>
#include <gtest/gtest-spi.h>

#include "app/MPIBootstrapPhase.h"
#include "planning/PlacementTopology.h"
#include "utils/MPISessionEnvironment.h"
#include "utils/MPISessionOwner.h"

#include <fcntl.h>
#include <sys/socket.h>
#include <sys/stat.h>
#include <sys/un.h>
#include <poll.h>
#include <signal.h>
#include <dirent.h>
#include <sys/resource.h>
#include <sys/syscall.h>
#include <sys/wait.h>
#include <unistd.h>

#include <cerrno>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <optional>
#include <set>
#include <string>
#include <vector>

#ifndef LLAMINAR_PLACEMENT_HELPER_PATH
#error "LLAMINAR_PLACEMENT_HELPER_PATH must be defined by the build"
#endif
#ifndef LLAMINAR_PLACEMENT_NATIVE_CHILD_PATH
#error "LLAMINAR_PLACEMENT_NATIVE_CHILD_PATH must be defined by the build"
#endif

namespace
{
    /// RAII environment mutation: every touched variable is restored on
    /// scope exit, on success AND on failure (no cross-test leakage).
    class ScopedEnvironment
    {
    public:
        void set(const char *key, const std::string &value)
        {
            remember(key);
            if (::setenv(key, value.c_str(), 1) != 0)
                FAIL() << "setenv(" << key << ") failed: " << std::strerror(errno);
        }
        void unset(const char *key)
        {
            remember(key);
            if (::unsetenv(key) != 0)
                FAIL() << "unsetenv(" << key << ") failed: " << std::strerror(errno);
        }
        ~ScopedEnvironment()
        {
            for (auto it = saved_.rbegin(); it != saved_.rend(); ++it)
            {
                if (it->second.empty()) ::unsetenv(it->first.c_str());
                else ::setenv(it->first.c_str(), it->second.c_str(), 1);
            }
        }
    private:
        void remember(const char *key)
        {
            if (seen_.count(key)) return;
            seen_.insert(key);
            const char *value = ::getenv(key);
            saved_.emplace_back(key, value ? value : "");
        }
        std::vector<std::pair<std::string, std::string>> saved_;
        std::set<std::string> seen_;
    };

    /// The actual admitted CPU set of THIS process (no hardcoded ids).
    std::vector<int> admittedCpus()
    {
        cpu_set_t mask;
        CPU_ZERO(&mask);
        if (sched_getaffinity(0, sizeof(mask), &mask) != 0)
            throw std::runtime_error("cannot read the test process affinity");
        std::vector<int> cpus;
        for (int cpu = 0; cpu < CPU_SETSIZE; ++cpu)
            if (CPU_ISSET(cpu, &mask)) cpus.push_back(cpu);
        return cpus;
    }

    /// EINTR-safe, error-checked wait; returns the raw wait status or -1
    /// with the exact errno captured for diagnosis.
    int checkedWait(const pid_t pid, int *wait_errno = nullptr)
    {
        if (pid < 0) { if (wait_errno) *wait_errno = EINVAL; return -1; }
        int status = 0;
        pid_t waited;
        do { waited = ::waitpid(pid, &status, 0); }
        while (waited < 0 && errno == EINTR);
        if (wait_errno) *wait_errno = waited == pid ? 0 : errno;
        if (waited != pid) return -1;
        return status;
    }

    int rawExit(const int status)
    {
        return WIFEXITED(status) ? WEXITSTATUS(status) : -1;
    }

    /// Checked env mutation for children (setenv errors abort the test).
    void checkedSet(const char *key, const std::string &value)
    {
        if (::setenv(key, value.c_str(), 1) != 0)
            FAIL() << "setenv(" << key << ") failed";
    }

    /// Run a command array; fork/exec failures are detectable (exec of a
    /// missing binary exits 127 in the child, never a false-pass 0).
    int run(const std::vector<std::string> &command)
    {
        const pid_t pid = ::fork();
        if (pid < 0) return -1;
        if (pid == 0)
        {
            std::vector<char *> argv;
            for (const auto &argument : command)
                argv.push_back(const_cast<char *>(argument.c_str()));
            argv.push_back(nullptr);
            ::execvp(command[0].c_str(), argv.data());
            std::_Exit(127); // exec failure is distinguishable from exit 0
        }
        return checkedWait(pid);
    }

    /// Managed mpirun chain with DERIVED geometry: ranks each mapped to
    /// pe admitted cpus; pe must divide the admitted set for disjoint ranks.
    int managedMpirun(const int ranks, const int pe, const std::string &session,
        ScopedEnvironment &env)
    {
        env.unset("OMP_NUM_THREADS");
        env.unset("OMP_PLACES");
        env.unset("OMP_PROC_BIND");
        env.set("LLAMINAR_MANAGED_PLACEMENT", "1");
        std::vector<std::string> command{"mpirun"};
        if (::geteuid() == 0) command.push_back("--allow-run-as-root");
        command.insert(command.end(),
            {"--bind-to", "none", "--map-by",
             "slot:PE=" + std::to_string(pe), "-np", std::to_string(ranks)});
        command.insert(command.end(),
            {LLAMINAR_PLACEMENT_HELPER_PATH, "--session=" + session, "--",
             LLAMINAR_PLACEMENT_NATIVE_CHILD_PATH});
        return run(command);
    }

}

TEST(PlacementIntegration, ManagedSingleRankFullBudgetPartition)
{
    const auto cpus = admittedCpus();
    ASSERT_GE(cpus.size(), 2u);
    ScopedEnvironment env;
    const int status = managedMpirun(1, static_cast<int>(cpus.size()),
        "integrate-1", env);
    EXPECT_EQ(rawExit(status), 0) << "raw " << status;
}

TEST(PlacementIntegration, ManagedDisjointTwoRanks)
{
    // Two ranks mapped slot:PE=half receive disjoint half-masks; each rank's
    // helper derives its own receipt and its admission verifies the native
    // place list against it. Rank identity comes from the MPI environment
    // per rank, correlated against the receipt.
    const auto cpus = admittedCpus();
    ASSERT_EQ(cpus.size() % 2, 0u) << "even split requires an even cpu count";
    ScopedEnvironment env;
    const int status = managedMpirun(2, static_cast<int>(cpus.size() / 2),
        "integrate-2", env);
    EXPECT_EQ(rawExit(status), 0) << "raw " << status;
}

TEST(PlacementIntegration, ManagedPartialBudgetLeavesIdlePlacesLegitimately)
{
    const auto cpus = admittedCpus();
    ASSERT_GE(cpus.size(), 4u);
    ScopedEnvironment env;
    const int status = managedMpirun(1, static_cast<int>(cpus.size()),
        "integrate-p", env);
    // Full allocation; the helper is authored a PARTIAL budget (a quarter of
    // the derived places, at least 1) — idle places are legitimate.
    // (Budget authored on the helper argv below.)
    std::vector<std::string> command{"mpirun"};
    if (::geteuid() == 0) command.push_back("--allow-run-as-root");
    command.insert(command.end(),
        {"--bind-to", "none", "--map-by",
         "slot:PE=" + std::to_string(cpus.size()), "-np", "1"});
    const int budget = std::max(1, static_cast<int>(cpus.size()) / 4);
    command.insert(command.end(),
        {LLAMINAR_PLACEMENT_HELPER_PATH, "--budget=" + std::to_string(budget),
         "--session=integrate-p", "--", LLAMINAR_PLACEMENT_NATIVE_CHILD_PATH});
    const int wired = run(command);
    EXPECT_EQ(rawExit(wired), 0) << "raw " << wired;
}

TEST(PlacementIntegration, NativeBoundLegalDuplicatePlacesAccepted)
{
    // Authored duplicate places are legal native grammar; the runtime binds
    // threads onto them; admission validates the authored list against the
    // native place list and must NOT reject duplication. Cpus DERIVED from
    // the actual admitted set (first two admitted ids).
    const auto cpus = admittedCpus();
    ASSERT_GE(cpus.size(), 2u);
    ScopedEnvironment env;
    env.unset("LLAMINAR_MANAGED_PLACEMENT");
    env.unset("LLAMINAR_PLACEMENT_RECEIPT_V1");
    env.set("OMP_PLACES", "{" + std::to_string(cpus[0]) + "},{" +
        std::to_string(cpus[0]) + "}");
    env.set("OMP_PROC_BIND", "true");
    env.set("OMP_NUM_THREADS", "2");
    const int status = run({LLAMINAR_PLACEMENT_NATIVE_CHILD_PATH});
    EXPECT_EQ(rawExit(status), 0) << "raw " << status;
}

TEST(PlacementIntegration, NativeBoundOversubscribedTeamHonored)
{
    const auto cpus = admittedCpus();
    ASSERT_GE(cpus.size(), 2u);
    ScopedEnvironment env;
    env.unset("LLAMINAR_MANAGED_PLACEMENT");
    env.unset("LLAMINAR_PLACEMENT_RECEIPT_V1");
    env.set("OMP_PLACES", "{" + std::to_string(cpus[0]) + "},{" +
        std::to_string(cpus[1]) + "}");
    env.set("OMP_PROC_BIND", "true");
    env.set("OMP_NUM_THREADS", "4"); // 4 threads over 2 places: user intent
    const int status = run({LLAMINAR_PLACEMENT_NATIVE_CHILD_PATH});
    EXPECT_EQ(rawExit(status), 0) << "raw " << status;
}

TEST(PlacementIntegration, NativeUnboundPolicyIsFirstClass)
{
    const auto cpus = admittedCpus();
    ASSERT_GE(cpus.size(), 2u);
    ScopedEnvironment env;
    env.unset("LLAMINAR_MANAGED_PLACEMENT");
    env.unset("LLAMINAR_PLACEMENT_RECEIPT_V1");
    env.set("OMP_PLACES", "{" + std::to_string(cpus[0]) + "},{" +
        std::to_string(cpus[1]) + "}");
    env.set("OMP_PROC_BIND", "false");
    env.set("OMP_NUM_THREADS", "2");
    const int status = run({LLAMINAR_PLACEMENT_NATIVE_CHILD_PATH});
    EXPECT_EQ(rawExit(status), 0) << "raw " << status;
}

TEST(PlacementIntegration, NativeMalformedAuthoredPlacesAreFatal)
{
    ScopedEnvironment env;
    env.unset("LLAMINAR_MANAGED_PLACEMENT");
    env.unset("LLAMINAR_PLACEMENT_RECEIPT_V1");
    env.set("OMP_PLACES", "{0},{1},baddata");
    env.set("OMP_PROC_BIND", "true");
    const int status = run({LLAMINAR_PLACEMENT_NATIVE_CHILD_PATH});
    EXPECT_EQ(rawExit(status), 1) << "raw " << status;
}

TEST(PlacementIntegration, ManagedMissingReceiptIsFatal)
{
    ScopedEnvironment env;
    env.set("LLAMINAR_MANAGED_PLACEMENT", "1");
    env.unset("LLAMINAR_PLACEMENT_RECEIPT_V1");
    const int status = run({LLAMINAR_PLACEMENT_NATIVE_CHILD_PATH});
    EXPECT_EQ(rawExit(status), 1) << "raw " << status;
}

TEST(PlacementIntegration, ManagedCorruptReceiptIsFatal)
{
    ScopedEnvironment env;
    env.set("LLAMINAR_MANAGED_PLACEMENT", "1");
    env.set("LLAMINAR_PLACEMENT_RECEIPT_V1",
        "v=1;session=s;host=h;rank=0;budget=1;budget_origin=AutoLocal;"
        "admitted=0;place_count=1;places={0};"
        "capacity_source=CpufreqKhz;capacity_values=1;checksum=deadbeef");
    const int status = run({LLAMINAR_PLACEMENT_NATIVE_CHILD_PATH});
    EXPECT_EQ(rawExit(status), 1) << "raw " << status;
}

TEST(PlacementIntegration, UnexpectedReceiptWithoutMarkerIsFatal)
{
    ScopedEnvironment env;
    env.unset("LLAMINAR_MANAGED_PLACEMENT");
    env.set("LLAMINAR_PLACEMENT_RECEIPT_V1", "garbage");
    const int status = run({LLAMINAR_PLACEMENT_NATIVE_CHILD_PATH});
    EXPECT_EQ(rawExit(status), 1) << "raw " << status;
}

TEST(PlacementIntegration, ValidReceiptWithHealthyOwnerPassesAndOwnerLossIsSpecificallyFatal)
{
    // POSITIVE CONTROL first: a VALID receipt (derived from THIS process's
    // actual mask, identity matching THIS host/rank env) with a HEALTHY
    // owner must PASS admission. Then ONLY the owner is removed: the
    // admission must fail with the SPECIFIC owner error — never a generic
    // rejection — and the directory rename is restored on ALL paths.
    const auto cpus = admittedCpus();
    ASSERT_GE(cpus.size(), 1u);
    auto root = llaminar2::MPISessionOwner::sharedDirectory();
    ASSERT_TRUE(root.has_value()) << "test process must run under the owner";
    ScopedEnvironment env;
    env.set("LLAMINAR_MANAGED_PLACEMENT", "1");
    const char *mpi_rank = ::getenv("OMPI_COMM_WORLD_RANK");
    const std::string rank = (mpi_rank && *mpi_rank) ? mpi_rank : "0";
    env.set("OMPI_COMM_WORLD_RANK", rank);
    auto receipt = llaminar2::placement::deriveLocalReceipt(cpus,
        "/sys/devices/system/cpu");
    llaminar2::placement::ReceiptSessionIdentity identity;
    identity.session = "owner-loss-probe";
    char host[256];
    ASSERT_EQ(::gethostname(host, sizeof(host)), 0);
    identity.host = host;
    identity.rank = rank;
    identity.budget = static_cast<long>(receipt.ordered_places().size());
    identity.budget_origin_explicit = false;
    const auto transport = llaminar2::placement::serializeReceiptTransport(
        receipt, identity);
    env.set("LLAMINAR_PLACEMENT_RECEIPT_V1", transport);
    // The helper exports the placement variables alongside the receipt; the
    // direct-child simulation must mirror them or the admission's bound-place
    // check would reject for the wrong reason.
    env.set("OMP_PLACES", receipt.omp_places_string());
    env.set("OMP_PROC_BIND", "close");
    env.set("OMP_NUM_THREADS", std::to_string(identity.budget));

    // POSITIVE CONTROL: healthy owner -> admission passes (exit 0).
    {
        const int status = run({LLAMINAR_PLACEMENT_NATIVE_CHILD_PATH});
        ASSERT_EQ(rawExit(status), 0) << "healthy-owner control must pass";
    }

    // OWNER LOSS: rename the owned directory away (RAII restoration below).
    const std::string dir_path = root->path();
    const std::string moved = dir_path + "-moved-" + std::to_string(::getpid());
    ASSERT_EQ(::rename(dir_path.c_str(), moved.c_str()), 0);
    struct Restore
    {
        ~Restore()
        {
            if (!restored_)
                ::rename(moved_.c_str(), original_.c_str());
        }
        std::string moved_, original_;
        bool restored_ = false;
    } restore{moved, dir_path};

    // The failure path: assert the SPECIFIC owner diagnostic over a pipe.
    {
        int pipe_fd[2];
        ASSERT_EQ(::pipe(pipe_fd), 0);
        const pid_t pid = ::fork();
        if (pid == 0)
        {
            ::close(pipe_fd[0]);
            ::dup2(pipe_fd[1], 2);
            std::vector<std::string> command{LLAMINAR_PLACEMENT_NATIVE_CHILD_PATH};
            std::vector<char *> argv;
            for (const auto &argument : command)
                argv.push_back(const_cast<char *>(argument.c_str()));
            argv.push_back(nullptr);
            ::execvp(command[0].c_str(), argv.data());
            std::_Exit(127);
        }
        ::close(pipe_fd[1]);
        char buffer[512] = {};
        const ssize_t got = ::read(pipe_fd[0], buffer, sizeof(buffer) - 1);
        ::close(pipe_fd[0]);
        const int status = checkedWait(pid);
        ASSERT_EQ(rawExit(status), 1) << "owner loss must be fatal";
        buffer[got > 0 ? got : 0] = '\0';
        // The SPECIFIC owner diagnostic, not a generic rejection.
        EXPECT_NE(std::strstr(buffer, "owner state is broken"), nullptr)
            << "diagnostic: " << buffer;
    }
    restore.restored_ = true;
    ASSERT_EQ(::rename(moved.c_str(), dir_path.c_str()), 0);

    // RESTORED: the healthy control passes again (owner restored cleanly).
    {
        const int status = run({LLAMINAR_PLACEMENT_NATIVE_CHILD_PATH});
        EXPECT_EQ(rawExit(status), 0) << "restored-owner control must pass";
    }
}

TEST(PlacementIntegration, RepeatedLaunchesRetireWithoutStrayRanks)
{
    // Real process-lifetime: three sequential managed launches; after each,
    // no process from THIS chain (session-token-scoped ownership, never
    // other jobs) may remain — the descendants retire under the existing
    // MPISessionOwner lifecycle.
    const auto cpus = admittedCpus();
    ASSERT_GE(cpus.size(), 2u);
    ScopedEnvironment env;
    // Ownership authority: the KERNEL's exact children file
    // (/proc/self/task/<tid>/children — the same source the session owner's
    // signalChildren uses). Only OUR direct children exist there; other jobs
    // never appear. The managed chain's direct child (mpirun) is waited by
    // checkedWait; any STILL-LIVE own child after the wait is a retirement
    // defect. (Grand-descendants are the mpirun/agent chain's own lifecycle
    // responsibility, verified by the chain's clean exit.)
    auto ownLiveChildren = [] {
        std::ifstream children("/proc/self/task/" +
            std::to_string(::getpid()) + "/children");
        std::vector<pid_t> live;
        pid_t child;
        while (children >> child) live.push_back(child);
        return live;
    };
    for (int round = 0; round < 3; ++round)
    {
        const std::string session = "retire-" + std::to_string(round);
        const int status = managedMpirun(1, static_cast<int>(cpus.size()),
            session, env);
        ASSERT_EQ(rawExit(status), 0) << "round " << round;
        const auto live = ownLiveChildren();
        EXPECT_TRUE(live.empty()) << "round " << round << ": " <<
            live.size() << " own child processes still live after the chain exit";
    }
}

TEST(PlacementIntegration, RepeatedLaunchesUnderOneSessionOwnerRetire)
{
    // Owned retirement: repeated managed launches under ONE session-owner
    // chain; the owner's actual wait (checkedWait) collects each chain, and
    // the shared-directory identity remains the SAME owner object throughout
    // (the session lifecycle outlives the launches).
    const auto cpus = admittedCpus();
    ASSERT_GE(cpus.size(), 2u);
    ScopedEnvironment env;
    auto root_before = llaminar2::MPISessionOwner::sharedDirectory();
    ASSERT_TRUE(root_before.has_value());
    const auto owner_identity = root_before->identity();
    for (int round = 0; round < 3; ++round)
    {
        const int status = managedMpirun(1, static_cast<int>(cpus.size()),
            "owned-" + std::to_string(round), env);
        ASSERT_EQ(rawExit(status), 0) << "round " << round;
    }
    auto root_after = llaminar2::MPISessionOwner::sharedDirectory();
    ASSERT_TRUE(root_after.has_value());
    EXPECT_EQ(root_after->identity(), owner_identity)
        << "the session owner's identity must be stable across launches";
}

namespace {

/// Spawn phases for the structured exact-handle spawn. Child ownership
/// transfers into the chain BEFORE the handle acquisition is attempted,
/// so an acquisition failure still leaves the LIVE child owned by the
/// chain (teardown re-acquires or fails visibly; it never leaks the
/// child and never fabricates success).
enum SpawnPhase
{
    SpawnChildExec = 0,      // the caller is the fork child: exec next
    SpawnFailed = -1,        // fork failed; nothing exists, nothing owned
    SpawnAcquireFailed = 2,  // child LIVE and chain-owned; handle missing
    SpawnedWithHandle = 1    // child live, owned, exact handle acquired
};

/// Scan gate: prove NO guardian root remains in the session dir. Any
/// remaining llaminar-mpi-<uid>- entry is a retirement failure (the
/// exact recorded identity or a new one — both fail; nothing else may
/// create guardian roots in the chain-exclusive dir); iterator and stat
/// errors fail. `failure` names the concrete residue on failure.
bool noGuardianRootsRemain(const std::string &session_dir,
    std::string &failure)
{
    std::error_code ec;
    for (const auto &entry :
        std::filesystem::directory_iterator(session_dir, ec))
    {
        const auto name = entry.path().filename().string();
        if (!name.starts_with("llaminar-mpi-" +
                std::to_string(::geteuid()) + "-"))
            continue;
        struct stat residue_stat {};
        const std::string identity = ::stat(entry.path().c_str(),
            &residue_stat) == 0
            ? " (dev=" + std::to_string(residue_stat.st_dev) + " ino=" +
                std::to_string(residue_stat.st_ino) + ")"
            : " (stat failed errno=" + std::to_string(errno) + ")";
        failure = "a guardian session root remains in the chain session "
                  "dir: " + name + identity;
        return false;
    }
    if (ec)
    {
        failure = "session-dir iteration failed: " + ec.message();
        return false;
    }
    return true;
}

/// Typed teardown outcome: a cleanup-only completion (the frontend was
/// collected by another waiter, so the native exit status was never
/// observed in-chain) is a DISTINCT state from a valid owned completion
/// — never conflated with success-by-death, never hidden inside a bool.
enum class TeardownStatus
{
    Completed,             // owned termination + exact in-chain collection
    CompletedCleanupOnly,  // retired, but the native status was collected elsewhere
    Failed                 // visible FATAL; handles/root/backing RETAINED
};

struct TeardownResult
{
    TeardownStatus status = TeardownStatus::Failed;
    bool reaped = false;              // exact in-chain collection happened
    bool collected_elsewhere = false; // wait ownership lost (visible note)
    bool has_exit_status = false;     // the native status is owned evidence
    int exit_status = 0;              // raw wait status when has_exit_status
    std::string report;
};

/// The bounded RAII owner for a cancellation-test chain. Constructed
/// BEFORE the first resource exists, so every acquisition failure still
/// runs teardown. Death is observed ONLY through the retained exact
/// handle's POLLIN exit notification; ECHILD (lost wait ownership) is a
/// separate visible state, never death evidence. Only an exact
/// in-chain waitpid collection (here or recorded by the test's own
/// bounded completion loop) sets `reaped`. On any teardown failure the
/// handles, the ack root AND the guardian session backing are RETAINED
/// (evidence) and the report is printed to stderr by the destructor.
/// A FAILED teardown leaves the chain retryable: the destructor performs
/// exactly one supervised retry (after any earlier-declared scoped
/// restoration ran), so an ASSERT path can never strand a live owned
/// chain. Ack storage (owner_root) and guardian backing (session_dir)
/// are SEPARATE directories: the guardian alone retires its roots, and
/// teardown never deletes live session roots as cleanup — the session
/// dir is removed only after a scan proves no guardian root remains.
struct CancellationChain
{
    pid_t frontend = -1;
    int frontend_pidfd = -1;
    int ack_socket = -1;
    /// The creator-retained grandchild handle TRANSFERRED via SCM_RIGHTS.
    int grandchild_pidfd = -1;
    /// Test-owned ack storage (socket + never-bound paths). NEVER hosts
    /// guardian roots.
    std::string owner_root;
    /// SEPARATE test-owned guardian backing (the chain's TMPDIR): the
    /// guardian's llaminar-mpi-<uid>-* roots land here exclusively.
    std::string session_dir;
    std::string ack_socket_path;
    bool reaped = false;
    bool collected_elsewhere = false;
    bool torn_down = false;
    bool has_exit_status = false;
    int exit_status = 0;
    std::string report;
    /// Stage bounds (production defaults; the focused tests shrink them).
    std::chrono::seconds term_bound{30}, kill_bound{10}, collect_bound{5};
    /// Optional out-param: the destructor's ONE supervised retry writes
    /// its result here when set, so a checked harness that aborts inside
    /// a helper scope can prove the ASSERT-path retirement AFTER the
    /// helper returned (the evidence outlives the chain).
    struct DestructorRetryEvidence
    {
        bool ran = false;
        TeardownResult result;
    };
    DestructorRetryEvidence *destructor_evidence = nullptr;

    /// Record an exact in-chain collection performed by the test's own
    /// bounded completion loop (the collected status is owned evidence).
    void recordOwnedCollection(const int raw_status)
    {
        reaped = true;
        has_exit_status = true;
        exit_status = raw_status;
    }

    /// Death oracle: POLLIN on the retained exact handle is the kernel's
    /// exit notification for THAT identity; ECHILD is never consulted.
    bool frontendDead() const
    {
        if (frontend_pidfd < 0) return false;
        pollfd p[] = {{frontend_pidfd, POLLIN, 0}};
        const int observed = ::poll(p, 1, 0);
        return observed == 1 && (p[0].revents & POLLIN);
    }

    bool awaitFrontendDeath(const std::chrono::seconds bound) const
    {
        const auto deadline = std::chrono::steady_clock::now() + bound;
        while (std::chrono::steady_clock::now() < deadline)
        {
            if (frontendDead()) return true;
            ::usleep(20000); // EINTR from a storm is harmless: retry
        }
        return frontendDead();
    }

    /// Bounded owned teardown on EVERY path, returning the typed
    /// outcome. Signals travel ONLY through the exact retained pidfd
    /// with CHECKED results; a missing handle first re-acquires through
    /// the same primitive on the still-owned unreaped child, and a
    /// failed re-acquisition is a VISIBLE FATAL (no numeric-pid
    /// fallback exists anywhere). Collection is EINTR-safe; only exact
    /// in-chain waitpid sets `reaped`; ECHILD is the visible
    /// lost-ownership state (CompletedCleanupOnly, without a native
    /// status). The ack root and the session dir are removed ONLY on a
    /// completed teardown whose session scan proves every guardian root
    /// retired; a Failed teardown retains everything.
    TeardownResult teardown()
    {
        TeardownResult result;
        bool ok = true;
        if (frontend > 0 && !reaped && !collected_elsewhere)
        {
            if (frontend_pidfd < 0)
            {
                // Real re-acquisition on the still-owned unreaped child
                // (identity still pinned). This is the same exact-handle
                // primitive — not a fallback, and never numeric.
                frontend_pidfd = static_cast<int>(
                    ::syscall(SYS_pidfd_open, frontend, 0));
                if (frontend_pidfd < 0)
                {
                    ok = false;
                    result.report += "FATAL: exact frontend handle could "
                        "not be (re)acquired at teardown errno=" +
                        std::to_string(errno) + " (no numeric-pid fallback "
                        "exists; the chain is left unterminated and owned "
                        "by the test process); ";
                }
            }
            if (ok)
            {
                if (!frontendDead())
                {
                    if (::syscall(SYS_pidfd_send_signal, frontend_pidfd,
                            SIGTERM, nullptr, 0) != 0 && errno != ESRCH)
                    {
                        ok = false;
                        result.report += "FATAL: pidfd SIGTERM failed "
                            "errno=" + std::to_string(errno) + "; ";
                    }
                    if (ok && !awaitFrontendDeath(term_bound))
                    {
                        result.report += "note: escalated to SIGKILL after "
                            "the SIGTERM bound; ";
                        if (::syscall(SYS_pidfd_send_signal, frontend_pidfd,
                                SIGKILL, nullptr, 0) != 0 && errno != ESRCH)
                        {
                            ok = false;
                            result.report += "FATAL: pidfd SIGKILL failed "
                                "errno=" + std::to_string(errno) + "; ";
                        }
                        if (ok && !awaitFrontendDeath(kill_bound))
                        {
                            ok = false;
                            result.report += "FATAL: the frontend survived "
                                "SIGKILL within the bound; ";
                        }
                    }
                }
                if (ok)
                {
                    // Bounded, EINTR-safe zombie collection. ONLY an
                    // exact successful in-chain waitpid sets `reaped`;
                    // EINTR retries; ECHILD is the visible lost-wait-
                    // ownership state (a cleanup-only completion WITHOUT
                    // the native status — a distinct typed outcome).
                    const auto collect_deadline =
                        std::chrono::steady_clock::now() + collect_bound;
                    for (;;)
                    {
                        int status = 0;
                        const pid_t waited =
                            ::waitpid(frontend, &status, WNOHANG);
                        if (waited == frontend)
                        {
                            reaped = true;
                            has_exit_status = true;
                            exit_status = status;
                            break;
                        }
                        if (waited < 0)
                        {
                            if (errno == EINTR) { /* retry in bound */ }
                            else if (errno == ECHILD)
                            {
                                collected_elsewhere = true;
                                result.report += "note: frontend wait "
                                    "ownership lost (collected elsewhere); "
                                    "the native exit status was not "
                                    "observed in-chain; ";
                                break;
                            }
                            else
                            {
                                ok = false;
                                result.report += "FATAL: waitpid errno=" +
                                    std::to_string(errno) + "; ";
                                break;
                            }
                        }
                        if (std::chrono::steady_clock::now() >
                            collect_deadline)
                        {
                            ok = false;
                            result.report += "FATAL: the frontend zombie "
                                "was not collected within the bound; ";
                            break;
                        }
                        ::usleep(10000);
                    }
                }
            }
        }
        if (ok && !session_dir.empty())
        {
            // GUARDIAN-BACKING GATE: the session dir is removed only when
            // a scan proves NO guardian root remains (the native guardian
            // alone retires its roots; iterator errors fail; live roots
            // are NEVER deleted as cleanup — a failed gate retains all
            // backing as evidence).
            std::string failure;
            if (!noGuardianRootsRemain(session_dir, failure))
            {
                ok = false;
                result.report += "FATAL: " + failure + "; ";
            }
        }
        if (ok)
        {
            if (collected_elsewhere)
                result.status = TeardownStatus::CompletedCleanupOnly;
            else
                result.status = TeardownStatus::Completed;
            // Completed: release everything. The ack socket and the two
            // test-owned directories are released only now, AFTER the
            // session gate passed (order matters: any earlier failure
            // retains them as evidence).
            if (frontend_pidfd >= 0) { ::close(frontend_pidfd); frontend_pidfd = -1; }
            if (grandchild_pidfd >= 0) { ::close(grandchild_pidfd); grandchild_pidfd = -1; }
            if (ack_socket >= 0) { ::close(ack_socket); ack_socket = -1; }
            if (!ack_socket_path.empty())
            {
                ::unlink(ack_socket_path.c_str());
                ack_socket_path.clear();
            }
            if (!session_dir.empty())
            {
                // EMPTY-DIRECTORY removal ONLY: rmdir fails ENOTEMPTY on
                // ANY residue, so unclassified backing is never erased —
                // residue is a Failed teardown with ALL evidence retained.
                if (::rmdir(session_dir.c_str()) != 0)
                {
                    result.status = TeardownStatus::Failed;
                    result.report += "FATAL: session-dir rmdir failed "
                        "errno=" + std::to_string(errno) + " (residue is "
                        "retained as evidence at: " + session_dir + "); ";
                }
                else
                    session_dir.clear();
            }
            if (result.status != TeardownStatus::Failed &&
                !owner_root.empty())
            {
                if (::rmdir(owner_root.c_str()) != 0)
                {
                    result.status = TeardownStatus::Failed;
                    result.report += "FATAL: ack-root rmdir failed errno=" +
                        std::to_string(errno) + " (residue is retained as "
                        "evidence at: " + owner_root + "); ";
                }
            }
        }
        if (result.status == TeardownStatus::Failed)
        {
            if (!owner_root.empty() || !session_dir.empty())
            {
                // Failure: the failed retirement's evidence is RETAINED —
                // handles stay open for a supervising retry, and BOTH the
                // ack root and the guardian session backing stay on disk
                // named with the attempt token.
                result.report += "RETAINED for evidence: ack root at: " +
                    (owner_root.empty() ? "<released>" : owner_root) +
                    "; session backing at: " +
                    (session_dir.empty() ? "<released>" : session_dir) +
                    "; ";
            }
            // A FAILED teardown stays RETRYABLE: torn_down is NOT set, so
            // the destructor performs exactly one supervised retry after
            // any earlier-declared scoped restoration (e.g. RLIMIT) has
            // run — an ASSERT path can never strand a live owned chain.
        }
        else
        {
            torn_down = true;
        }
        result.reaped = reaped;
        result.collected_elsewhere = collected_elsewhere;
        result.has_exit_status = has_exit_status;
        result.exit_status = exit_status;
        report += result.report;
        return result;
    }

    /// The report is ALWAYS visible: ASSERT paths return before the test
    /// body can observe it, so the destructor prints it to stderr
    /// (captured in the test log) whenever a teardown did not cleanly
    /// complete or left notes. If an explicit teardown FAILED, the
    /// destructor performs exactly ONE supervised retry (recoverable
    /// failure: earlier-declared scoped restorations have run by now)
    /// and marks the test failed if that retry does not complete.
    ~CancellationChain()
    {
        if (!torn_down)
        {
            const auto retry = teardown();
            if (destructor_evidence != nullptr)
            {
                destructor_evidence->ran = true;
                destructor_evidence->result = retry;
            }
            if (retry.status == TeardownStatus::Failed)
                ADD_FAILURE() << "destructor retirement retry failed: "
                              << retry.report;
        }
        if (!report.empty())
            std::fprintf(stderr, "[cancellation-chain] teardown report: %s\n",
                report.c_str());
    }
};

/// Fork and acquire the EXACT handle immediately: pidfd_open on the
/// unreaped direct child (the existing owner primitive signalChildren
/// uses). The child's identity is pinned for its whole unreaped
/// lifetime — even an instant exit leaves a zombie whose pid cannot be
/// reused before this parent reaps it — so no exit/reap/reuse window
/// exists. A failed acquisition returns SpawnAcquireFailed with the
/// live child already recorded in the chain (never resolved by a
/// numeric-pid fallback; those are absent everywhere).
int spawnFrontendWithExactHandle(CancellationChain &chain)
{
    const pid_t child = ::fork();
    if (child < 0) return SpawnFailed;
    if (child == 0) return SpawnChildExec;
    chain.frontend = child; // OWNERSHIP FIRST: the chain owns the child
    chain.frontend_pidfd = static_cast<int>(
        ::syscall(SYS_pidfd_open, child, 0));
    return chain.frontend_pidfd >= 0 ? SpawnedWithHandle : SpawnAcquireFailed;
}

/// Scoped RLIMIT_NOFILE exhaustion for the REAL acquisition-failure
/// injection. RLIMIT_NOFILE caps the next allocatable fd NUMBER, not the
/// open count (a gappy fd table defeats count-based bounds): opendir
/// takes the lowest free slot, so its dirfd IS the next allocation after
/// it is closed; lowering the soft limit to exactly that number makes
/// the next fd allocation fail EMFILE.
///
/// SAFETY (review finding): restoration is owned by this RAII scope and
/// ordered BEFORE any later-declared chain cleanup — destruction is
/// reverse-declaration, so declaring this AFTER the chain/ack owners
/// guarantees the limit is restored before the chain's supervised
/// teardown on EVERY path, including ASSERT aborts. The destructor
/// VERIFIES the restoration (non-fatal ADD_FAILURE), so a leak is a
/// visible test failure, never a silent process-wide throttle.
struct ScopedNoFileExhaustion
{
    rlimit saved {};
    bool armed = false;

    bool arm()
    {
        if (armed) return false;
        if (::getrlimit(RLIMIT_NOFILE, &saved) != 0) return false;
        int next_free = -1;
        {
            ::DIR *dir = ::opendir("/proc/self/fd");
            if (dir == nullptr) return false;
            next_free = ::dirfd(dir);
            ::closedir(dir);
        }
        if (next_free <= 0) return false;
        rlimit lowered = saved;
        lowered.rlim_cur = static_cast<rlim_t>(next_free);
        if (lowered.rlim_cur > saved.rlim_max) return false;
        if (::setrlimit(RLIMIT_NOFILE, &lowered) != 0) return false;
        armed = true;
        return true;
    }

    /// Explicit early restoration (idempotent; the supervised-recovery
    /// tests restore before retrying teardown).
    void restore_now()
    {
        if (!armed) return;
        armed = false;
        if (::setrlimit(RLIMIT_NOFILE, &saved) != 0)
            ADD_FAILURE() << "RLIMIT_NOFILE early restoration failed";
    }

    ~ScopedNoFileExhaustion()
    {
        if (!armed) return;
        armed = false;
        if (::setrlimit(RLIMIT_NOFILE, &saved) != 0)
        {
            ADD_FAILURE() << "RLIMIT_NOFILE restoration failed";
            return;
        }
        rlimit now {};
        if (::getrlimit(RLIMIT_NOFILE, &now) != 0 ||
            now.rlim_cur != saved.rlim_cur)
            ADD_FAILURE() << "RLIMIT_NOFILE not actually restored";
    }
};

/// Spawn with a REAL exact-handle acquisition failure under an ARMED
/// ScopedNoFileExhaustion (the caller owns arming and restoration). The
/// child restores its inherited limit before exec. Returns the observed
/// acquisition errno (EMFILE expected), 0 if the injection did not bite
/// (the handle was honestly acquired), or -1 on spawn failure (nothing
/// exists, nothing owned).
template <typename ExecChild>
int spawnFrontendWithForcedAcquisitionFailure(CancellationChain &chain,
    const ScopedNoFileExhaustion &limit, ExecChild &&exec_child)
{
    const pid_t child = ::fork();
    if (child < 0) return -1;
    if (child == 0)
    {
        rlimit unthrottled = limit.saved;
        // A failed restoration must be VISIBLE, never a silently
        // throttled exec chain (distinct exit code 125).
        if (::setrlimit(RLIMIT_NOFILE, &unthrottled) != 0)
            std::_Exit(125);
        exec_child();
        std::_Exit(126); // unreachable if exec_child execs
    }
    chain.frontend = child; // ownership first, exactly as the normal spawn
    chain.frontend_pidfd = static_cast<int>(
        ::syscall(SYS_pidfd_open, child, 0));
    if (chain.frontend_pidfd >= 0) return 0; // injection did not bite
    return errno;
}

/// Strict decimal parse: digits only, at least one digit, full
/// consumption, ERANGE and bounds checked.
bool parseFrameValue(const std::string &digits, const long max_value,
    long &out)
{
    if (digits.empty() || digits.size() > 18) return false;
    for (const char c : digits)
        if (c < '0' || c > '9') return false;
    errno = 0;
    char *end = nullptr;
    const long value = std::strtol(digits.c_str(), &end, 10);
    if (errno == ERANGE || end != digits.c_str() + digits.size()) return false;
    if (value < 0 || value > max_value) return false;
    out = value;
    return true;
}

/// Receive ONE ack datagram with its optional transferred SCM_RIGHTS
/// handle, bounded by the caller's deadline (EINTR-safe).
bool receiveAck(const int ack_socket, std::string &frame, int &transferred_fd,
    const std::chrono::steady_clock::time_point &deadline)
{
    transferred_fd = -1;
    while (std::chrono::steady_clock::now() < deadline)
    {
        pollfd p[] = {{ack_socket, POLLIN, 0}};
        const int polled = ::poll(p, 1, 250);
        if (polled < 0)
        {
            if (errno == EINTR) continue;
            return false;
        }
        if (polled == 0) continue;
        char buffer[256] = {};
        char control[CMSG_SPACE(sizeof(int))] = {};
        iovec vector { buffer, sizeof(buffer) - 1 };
        msghdr message {};
        message.msg_iov = &vector;
        message.msg_iovlen = 1;
        message.msg_control = control;
        message.msg_controllen = sizeof(control);
        const ssize_t n = ::recvmsg(ack_socket, &message, MSG_CMSG_CLOEXEC);
        if (n < 0)
        {
            if (errno == EINTR || errno == EAGAIN) continue;
            return false;
        }
        if (n == 0) continue;
        for (cmsghdr *header = CMSG_FIRSTHDR(&message); header;
             header = CMSG_NXTHDR(&message, header))
            if (header->cmsg_level == SOL_SOCKET &&
                header->cmsg_type == SCM_RIGHTS &&
                header->cmsg_len >= CMSG_LEN(sizeof(int)))
                std::memcpy(&transferred_fd, CMSG_DATA(header), sizeof(int));
        frame.assign(buffer, static_cast<std::size_t>(n));
        return true;
    }
    return false;
}

/// Strict frame classifier: exactly "<tag> <attempt> <digits>\n".
bool matchAckFrame(const std::string &frame, const std::string &tag,
    const std::string &attempt, const long max_value, long &value)
{
    const std::string prefix = tag + " " + attempt + " ";
    if (!frame.starts_with(prefix)) return false;
    if (frame.empty() || frame.back() != '\n') return false;
    const std::string digits =
        frame.substr(prefix.size(), frame.size() - prefix.size() - 1);
    return parseFrameValue(digits, max_value, value);
}

/// One chain-scoped setup with IMMEDIATE, EXCLUSIVELY-TRACKED ownership.
/// Every acquisition is recorded the moment it SUCCEEDS exclusively:
/// - a mkdir that fails (EEXIST included) means the path was NEVER owned
///   and dispose() never touches it (a pre-existing directory with
///   unknown contents is never removed);
/// - the ack root and the guardian session backing are SEPARATE sibling
///   directories, so ack-storage cleanup can never delete guardian
///   roots;
/// - a failure at ANY stage disposes exactly the successfully acquired
///   set (no leaks, no foreign deletions), and release() transfers the
///   completed set to the chain (whose teardown then owns it).
struct AckSetup
{
    std::string attempt, owner_root, ack_socket_path, session_dir;
    int ack_socket = -1;
    bool root_created = false, session_created = false, socket_bound = false;
    bool released = false;

    ~AckSetup() { dispose(); }

    /// Dispose EXACTLY the exclusively-acquired set. Attempted-but-failed
    /// or pre-existing paths are never owned and never removed.
    void dispose()
    {
        if (released) return;
        if (ack_socket >= 0) { ::close(ack_socket); ack_socket = -1; }
        if (socket_bound && !ack_socket_path.empty())
            ::unlink(ack_socket_path.c_str());
        ack_socket_path.clear();
        socket_bound = false;
        if (session_created && !session_dir.empty())
        {
            std::error_code ec;
            std::filesystem::remove_all(session_dir, ec);
        }
        session_created = false;
        session_dir.clear();
        if (root_created && !owner_root.empty())
        {
            std::error_code ec;
            std::filesystem::remove_all(owner_root, ec);
        }
        root_created = false;
        owner_root.clear();
    }

    /// Transfer ownership of the completed set to the chain.
    void release() { released = true; }

    bool create(const char *prefix, const bool with_session_dir,
        const char *explicit_attempt = nullptr)
    {
        attempt = explicit_attempt
            ? std::string(explicit_attempt)
            : std::string(prefix) + "-" + std::to_string(::getpid()) + "-" +
                std::to_string(std::chrono::steady_clock::now()
                    .time_since_epoch().count());
        owner_root = "/tmp/placement-ack-owner-" + attempt;
        // EXCLUSIVE acquisition: EEXIST (or any failure) means this setup
        // never owned the path — dispose() must not remove it.
        if (::mkdir(owner_root.c_str(), 0700) != 0) { dispose(); return false; }
        root_created = true;
        struct stat root_stat {};
        if (::stat(owner_root.c_str(), &root_stat) != 0 ||
            (root_stat.st_mode & 0777) != 0700u)
        { dispose(); return false; }
        if (with_session_dir)
        {
            // SEPARATE sibling directory: guardian backing is never
            // nested inside ack storage.
            session_dir = "/tmp/placement-ack-session-" + attempt;
            if (::mkdir(session_dir.c_str(), 0700) != 0)
            { dispose(); return false; }
            session_created = true;
        }
        ack_socket_path = owner_root + "/ack.sock";
        ack_socket = ::socket(AF_UNIX, SOCK_DGRAM | SOCK_CLOEXEC, 0);
        if (ack_socket < 0) { dispose(); return false; }
        sockaddr_un address {};
        address.sun_family = AF_UNIX;
        std::strncpy(address.sun_path, ack_socket_path.c_str(),
            sizeof(address.sun_path) - 1);
        if (::bind(ack_socket, reinterpret_cast<sockaddr *>(&address),
                sizeof(address)) != 0)
        { dispose(); return false; }
        socket_bound = true;
        return true;
    }
};

/// Transfer the completed ack setup into the chain (single ownership).
void adoptAckSetup(CancellationChain &chain, AckSetup &ack)
{
    chain.ack_socket = ack.ack_socket;
    chain.ack_socket_path = ack.ack_socket_path;
    chain.owner_root = ack.owner_root;
    chain.session_dir = ack.session_dir;
    ack.ack_socket = -1; // the chain owns the fd now
    ack.release();       // the AckSetup destructor must not double-free
}

/// The retained identity of ONE live guardian session root: the path
/// AND an OPEN DIRECTORY HANDLE (O_DIRECTORY|O_NOFOLLOW) whose fstat
/// IS the identity authority — device, inode, and the nlink-based
/// retirement observation all derive from the SAME handle that pins
/// the EXACT original inode through retirement, so no path-metadata
/// race exists between observation and pinning. fstat on the retained
/// handle observes st_nlink==0 once the original directory is truly
/// unlinked; a renamed or moved root keeps st_nlink>0 and fails the
/// retirement check. A same-name replacement is a different inode and
/// is rejected by the scan against the retained identity.
struct RetainedRootHandle
{
    std::string path;
    dev_t device = 0;
    ino_t inode = 0;
    int directory_fd = -1;

    RetainedRootHandle() = default;
    RetainedRootHandle(std::string p, const dev_t d, const ino_t i,
        const int fd)
        : path(std::move(p)), device(d), inode(i), directory_fd(fd) {}
    RetainedRootHandle(const RetainedRootHandle &) = delete;
    RetainedRootHandle &operator=(const RetainedRootHandle &) = delete;
    RetainedRootHandle(RetainedRootHandle &&other) noexcept
        : path(std::move(other.path)), device(other.device),
          inode(other.inode), directory_fd(other.directory_fd)
    { other.directory_fd = -1; }
    RetainedRootHandle &operator=(RetainedRootHandle &&other) noexcept
    {
        if (this != &other)
        {
            if (directory_fd >= 0) ::close(directory_fd);
            path = std::move(other.path);
            device = other.device;
            inode = other.inode;
            directory_fd = other.directory_fd;
            other.directory_fd = -1;
        }
        return *this;
    }
    ~RetainedRootHandle()
    {
        if (directory_fd >= 0) ::close(directory_fd);
    }

    /// True when the ORIGINAL directory inode is no longer linked
    /// anywhere (truly retired; renames/moves keep st_nlink>0 and fail).
    /// fstat errors are failures, never evidence of retirement.
    bool originalUnlinked(std::string &failure) const
    {
        if (directory_fd < 0)
        { failure = "no retained directory handle for: " + path; return false; }
        struct stat handle_stat {};
        if (::fstat(directory_fd, &handle_stat) != 0)
        {
            failure = "fstat on the retained handle failed for: " + path +
                " errno=" + std::to_string(errno);
            return false;
        }
        if (handle_stat.st_nlink != 0)
        {
            failure = "the original root directory is still linked "
                "(renamed or moved, not retired): " + path;
            return false;
        }
        return true;
    }
};

/// Observe the guardian's LIVE session roots in the chain-exclusive
/// session directory: at least one must exist (proof this chain's
/// isolated dir actually hosted real guardian roots — a later empty
/// observation can never pass vacuously); directory-iterator errors AND
/// per-entry stat errors are FAILED observations (nullopt), never
/// silent skips; every captured root carries its stat identity.
std::optional<std::vector<RetainedRootHandle>> observeLiveSessionRoots(
    const std::string &session_dir)
{
    std::error_code ec;
    std::vector<RetainedRootHandle> roots;
    for (const auto &entry :
        std::filesystem::directory_iterator(session_dir, ec))
    {
        const auto name = entry.path().filename().string();
        if (!name.starts_with("llaminar-mpi-" +
                std::to_string(::geteuid()) + "-"))
            continue;
        // Open the directory handle FIRST (O_NOFOLLOW: a guardian-named
        // SYMLINK cannot impersonate a root), then derive the identity
        // FROM fstat ON THE HANDLE — the opened fd is the single identity
        // authority, so no stat-then-open metadata/handle race exists
        // (the reviewer's hardening: stat(path) + independent open could
        // observe one inode and pin another).
        const int handle = ::open(entry.path().c_str(),
            O_RDONLY | O_DIRECTORY | O_NOFOLLOW | O_CLOEXEC);
        if (handle < 0)
            return std::nullopt; // open errors FAIL, never hide
        struct stat handle_stat {};
        if (::fstat(handle, &handle_stat) != 0)
        {
            ::close(handle);
            return std::nullopt; // fstat errors FAIL, never hide
        }
        if (!S_ISDIR(handle_stat.st_mode))
        {
            ::close(handle);
            return std::nullopt; // not a directory: FAIL, never hide
        }
        roots.emplace_back(entry.path().string(), handle_stat.st_dev,
            handle_stat.st_ino, handle);
    }
    if (ec) return std::nullopt; // iterator errors FAIL, never hide
    return roots;
}

/// Validate TRUE retirement of the authenticated live identities: no
/// llaminar-mpi-<uid>- entry may remain, and any same-name directory is
/// checked against the captured stat identities (the exact recorded
/// root surviving is the primary failure; a DIFFERENT identity at the
/// same name is equally disallowed — nothing else may create guardian
/// roots in the chain-exclusive dir). Iterator/stat errors fail. This
/// validates observed retirement; it never erases evidence (removal is
/// teardown's separate, gated step).
bool validateRootRetirement(const std::vector<RetainedRootHandle> &live,
    const std::string &session_dir, std::string &failure)
{
    // Scan: no guardian-named entry may remain; any survivor's stat
    // identity is COMPARED against the retained handles (the exact
    // recorded root surviving is named as such; a different identity at
    // a guardian name — including a same-name replacement — is equally
    // rejected).
    std::error_code ec;
    for (const auto &entry :
        std::filesystem::directory_iterator(session_dir, ec))
    {
        const auto name = entry.path().filename().string();
        if (!name.starts_with("llaminar-mpi-" +
                std::to_string(::geteuid()) + "-"))
            continue;
        struct stat residue_stat {};
        if (::stat(entry.path().c_str(), &residue_stat) != 0)
        {
            failure = "stat failed on remaining guardian root: " + name;
            return false;
        }
        bool is_recorded_identity = false;
        for (const auto &retained : live)
            if (retained.device == residue_stat.st_dev &&
                retained.inode == residue_stat.st_ino)
                is_recorded_identity = true;
        failure = (is_recorded_identity
            ? "the exact recorded root survived retirement: "
            : "an unexpected guardian root is present (same-name "
              "replacement or new root): ") + name;
        return false;
    }
    if (ec)
    {
        failure = "session-dir iteration failed: " + ec.message();
        return false;
    }
    // Retained-handle authority: every ORIGINAL directory inode must be
    // truly unlinked (st_nlink==0 through the open handle) — a rename or
    // move keeps the original inode linked and fails here even though no
    // guardian-NAMED entry remains.
    for (const auto &retained : live)
        if (!retained.originalUnlinked(failure))
            return false;
    return true;
}

/// Post-teardown identity check for teardown-driven retirement: the
/// teardown's internal gate already proved no guardian root remained
/// before removing the session dir; here every recorded stat identity's
/// path must no longer resolve (the exact captured directories are gone).
bool recordedIdentitiesGone(const std::vector<RetainedRootHandle> &live)
{
    std::string failure;
    for (const auto &retained : live)
        if (!retained.originalUnlinked(failure))
            return false;
    return true;
}

/**
 * @brief Compose the owner, MPI, and diagnostic helper argument boundaries.
 * @param cpus Observed admissible CPU set for the diagnostic launch.
 * @param session Opaque diagnostic label; never an authenticated MPI identity.
 * @return Exact argv with three independent delimiters.
 */
std::vector<std::string> ownedPlacementDiagnosticCommand(const std::vector<int> &cpus,
    const std::string &session)
{
    std::vector<std::string> command{LLAMINAR_MPI_SESSION_PATH, "--", "mpirun"};
    if (::geteuid() == 0) command.push_back("--allow-run-as-root");
    // The owner must stop inspecting MPI options at MPI's own delimiter,
    // not at the later delimiter between the diagnostic helper and its child.
    command.insert(command.end(),
        {"--bind-to", "none", "--map-by",
         "slot:PE=" + std::to_string(cpus.size()), "-np", "1", "--",
         LLAMINAR_PLACEMENT_HELPER_PATH, "--session=" + session, "--",
         LLAMINAR_PLACEMENT_NATIVE_CHILD_PATH});
    return command;
}

/// Exec the production session owner around an isolated placement diagnostic.
void execProductionFrontend(const std::vector<int> &cpus,
    const std::string &session)
{
    const auto command = ownedPlacementDiagnosticCommand(cpus, session);
    std::vector<char *> argv;
    for (const auto &argument : command)
        argv.push_back(const_cast<char *>(argument.c_str()));
    argv.push_back(nullptr);
    ::execv(command[0].c_str(), argv.data());
    std::_Exit(127); // exec failure is distinguishable from other exits
}

/// Shared readiness wait: the rank's RDY record, strictly parsed,
/// bounded. Returns false on deadline (no frames invented).
bool awaitRankReady(CancellationChain &chain, const AckSetup &ack,
    const std::chrono::seconds bound)
{
    const auto deadline = std::chrono::steady_clock::now() + bound;
    long ack_rank = -1;
    for (;;)
    {
        std::string frame;
        int transferred = -1;
        if (!receiveAck(chain.ack_socket, frame, transferred, deadline))
            return false;
        if (matchAckFrame(frame, "RDY", ack.attempt, 1 << 20, ack_rank))
        {
            if (transferred >= 0) ::close(transferred);
            return ack_rank == 0;
        }
        if (transferred >= 0) ::close(transferred);
        // A GC record (grandchild) may arrive first in some orders; keep
        // waiting within the bound for the RDY.
        if (std::chrono::steady_clock::now() >= deadline) return false;
    }
}

} // namespace

TEST(PlacementIntegration, OwnedDiagnosticKeepsAllThreeArgumentBoundaries)
{
    const auto actual = ownedPlacementDiagnosticCommand({2, 7}, "fixture label");
    std::vector<std::string> expected{LLAMINAR_MPI_SESSION_PATH, "--", "mpirun"};
    if (::geteuid() == 0) expected.push_back("--allow-run-as-root");
    expected.insert(expected.end(), {"--bind-to", "none", "--map-by", "slot:PE=2",
        "-np", "1", "--", LLAMINAR_PLACEMENT_HELPER_PATH,
        "--session=fixture label", "--", LLAMINAR_PLACEMENT_NATIVE_CHILD_PATH});
    EXPECT_EQ(actual, expected);
}

TEST(PlacementIntegration, AckSetupCollisionLeavesPreexistingPathUntouched)
{
    // COLLISION NEGATIVE (review finding): a pre-existing directory at
    // the exact attempted ack-root path must NEVER be deleted by the
    // failed setup. mkdir EEXIST means the path was never owned;
    // dispose() removes only exclusively-acquired paths. The sentinel
    // contents prove the foreign directory is untouched, byte for byte.
    const std::string foreign =
        "/tmp/placement-ack-owner-collide-" + std::to_string(::getpid());
    const std::string sentinel = foreign + "/sentinel.payload";
    ASSERT_EQ(::mkdir(foreign.c_str(), 0700), 0) << "fixture mkdir";
    {
        std::ofstream file(sentinel);
        ASSERT_TRUE(file.is_open());
        file << "pre-existing foreign content " << ::getpid() << "\n";
    }
    std::string sentinel_before;
    {
        std::ifstream file(sentinel);
        std::getline(file, sentinel_before);
    }

    {
        AckSetup ack;
        // The exact collision: create() attempts the SAME path.
        EXPECT_FALSE(ack.create("collide", false,
            ("collide-" + std::to_string(::getpid())).c_str()))
            << "EEXIST must fail the setup";
        // The setup owns nothing; its destructor must not have removed
        // the foreign tree.
        EXPECT_TRUE(std::filesystem::exists(foreign));
    }

    // The foreign directory and its sentinel survive byte-identical.
    ASSERT_TRUE(std::filesystem::exists(sentinel));
    std::string sentinel_after;
    {
        std::ifstream file(sentinel);
        std::getline(file, sentinel_after);
    }
    EXPECT_EQ(sentinel_before, sentinel_after)
        << "pre-existing contents must be untouched";
    // Clean up this test's own fixture debris (it is test-created).
    std::error_code ec;
    std::filesystem::remove_all(foreign, ec);
    ASSERT_FALSE(ec);
    ASSERT_FALSE(std::filesystem::exists(foreign));
}

TEST(PlacementIntegration, RootRetirementValidationControlsCatchRenamedAndReplacedRoots)
{
    // CONTROLS for the authenticated root-retirement validation (review
    // finding): a RENAMED live root (still guardian-named, original inode
    // alive) and a SAME-NAME REPLACEMENT (original inode gone, a new
    // directory at the same path) must BOTH fail validation, while TRUE
    // retirement (original inode unlinked, nothing guardian-named
    // remaining) passes. The retained directory handles (fstat nlink)
    // and the identity-comparing scan are the authorities — never bare
    // path/prefix absence.
    const std::string base =
        "/tmp/placement-ack-session-ctrl-" + std::to_string(::getpid());
    ASSERT_EQ(::mkdir(base.c_str(), 0700), 0) << "fixture mkdir";
    const std::string root_name =
        "llaminar-mpi-" + std::to_string(::geteuid()) + "-ctrl0001";
    const std::string root = base + "/" + root_name;
    const std::string renamed = base + "/" + root_name + "-renamed";
    ASSERT_EQ(::mkdir(root.c_str(), 0700), 0) << "fixture root mkdir";
    {
        // A live root needs content parity with real guardian roots
        // (they contain files); the validation only inspects identity.
        std::ofstream file(root + "/inner");
        ASSERT_TRUE(file.is_open());
    }

    // Retain the live root's identity (stat + open directory handle).
    const auto live = observeLiveSessionRoots(base);
    ASSERT_TRUE(live.has_value())
        << "observation failed (iterator/stat/open error)";
    ASSERT_EQ(live->size(), 1u);
    EXPECT_EQ(live->front().path, root);

    std::string failure;

    // CONTROL A — RENAMED live root: the original inode survives under a
    // different (still guardian-named) path. The scan must reject the
    // renamed entry, and the retained handle must report the original
    // inode still linked (nlink > 0).
    ASSERT_EQ(::rename(root.c_str(), renamed.c_str()), 0);
    EXPECT_FALSE(validateRootRetirement(*live, base, failure))
        << "a renamed live root must fail validation";
    EXPECT_NE(failure.find("renamed"), std::string::npos)
        << "the retained handle must identify the rename: " << failure;
    failure.clear();
    EXPECT_FALSE(live->front().originalUnlinked(failure))
        << "the retained handle must observe nlink>0 for a renamed root";
    EXPECT_NE(failure.find("still linked"), std::string::npos) << failure;
    // Restore for the next control.
    ASSERT_EQ(::rename(renamed.c_str(), root.c_str()), 0);

    // CONTROL B — SAME-NAME REPLACEMENT: remove the original directory,
    // then create a NEW directory at the same path. The original inode
    // is truly gone, but a guardian-named entry remains — validation
    // must fail (the scan rejects the different-identity entry).
    ASSERT_EQ(::unlink((root + "/inner").c_str()), 0);
    ASSERT_EQ(::rmdir(root.c_str()), 0);
    ASSERT_EQ(::mkdir(root.c_str(), 0700), 0);
    EXPECT_FALSE(validateRootRetirement(*live, base, failure))
        << "a same-name replacement root must fail validation";
    EXPECT_NE(failure.find("unexpected guardian root"), std::string::npos)
        << "the scan must name the replacement: " << failure;
    failure.clear();

    // CONTROL C — TRUE retirement: remove the replacement; validation
    // passes (no guardian-named entry; every retained original inode
    // unlinked with nlink==0 through the handles).
    ASSERT_EQ(::rmdir(root.c_str()), 0);
    EXPECT_TRUE(validateRootRetirement(*live, base, failure)) << failure;
    EXPECT_TRUE(live->front().originalUnlinked(failure))
        << "the retained handle must observe nlink==0 after true "
           "retirement: " << failure;

    // Fixture cleanup (empty rmdir; any residue fails loudly).
    ASSERT_EQ(::rmdir(base.c_str()), 0) << "fixture cleanup";
}

TEST(PlacementIntegration, SigtermCancelsLiveChainThroughOwnerProtocol)
{
    // The COMPLETE production cancellation path through the REAL frontend
    // (llaminar_mpi_session: guardian subreaper + Signal-packet control
    // channel + exact-identity signalChildren). The ack channel is a unix
    // datagram socket BY PATH in a unique 0700 root; both records are
    // mandatory and strictly parsed; the grandchild identity is the
    // CREATOR-RETAINED pidfd transferred via SCM_RIGHTS; surviving-parent
    // adoption is evidenced by the sleeper's ADOPTED record before its
    // POLLIN death on the SAME transferred handle; the guardian's session
    // roots are isolated into this chain's OWN session directory (a
    // SEPARATE sibling of the ack root) and authenticated by STAT
    // IDENTITY (live identities observed during readiness must be truly
    // retired — never a vacuous path-only check); and the frontend's exit
    // status is the pinned propagated cancellation status, recorded as
    // owned in-chain evidence.
    const auto cpus = admittedCpus();
    ASSERT_GE(cpus.size(), 2u);
    ScopedEnvironment env;
    env.unset("OMP_NUM_THREADS");
    env.unset("OMP_PLACES");
    env.unset("OMP_PROC_BIND");
    env.set("LLAMINAR_MANAGED_PLACEMENT", "1");
    env.set("LLAMINAR_NATIVE_CHILD_PARK", "1");
    env.set("LLAMINAR_NATIVE_CHILD_GRANDCHILD", "1");

    CancellationChain chain;
    AckSetup ack;
    ASSERT_TRUE(ack.create("cancel", true))
        << "ack setup failed errno=" << errno;
    adoptAckSetup(chain, ack);
    const std::string session_dir = chain.session_dir;
    // The chain's session roots are isolated inside this test-owned 0700
    // directory: the guardian's sessionBase() (TMPDIR) points here, so
    // ONLY this guardian's llaminar-mpi-<uid>-* roots can appear in it.
    env.set("TMPDIR", session_dir);
    env.set("LLAMINAR_NATIVE_CHILD_ACK_SOCK", chain.ack_socket_path);
    env.set("LLAMINAR_NATIVE_CHILD_ATTEMPT", ack.attempt);

    const int phase = spawnFrontendWithExactHandle(chain);
    if (phase == SpawnChildExec) execProductionFrontend(cpus, ack.attempt);
    ASSERT_EQ(phase, SpawnedWithHandle)
        << "spawn/acquisition failed errno=" << errno;

    // The bounded readiness: BOTH records mandatory, strict frames; the GC
    // record must carry the transferred creator-retained handle.
    const auto ready_deadline = std::chrono::steady_clock::now() +
        std::chrono::seconds(30);
    bool have_ready = false, have_grandchild = false;
    long ack_rank = -1, reported_pid = -1;
    while (!(have_ready && have_grandchild))
    {
        std::string frame;
        int transferred = -1;
        if (!receiveAck(chain.ack_socket, frame, transferred, ready_deadline))
            FAIL() << "readiness deadline exceeded (RDY or GC missing)";
        if (matchAckFrame(frame, "RDY", ack.attempt, 1 << 20, ack_rank))
        {
            ASSERT_EQ(transferred, -1)
                << "RDY must not carry a handle [" << frame << "]";
            have_ready = true;
        }
        else if (matchAckFrame(frame, "GC", ack.attempt, 2000000000L,
            reported_pid))
        {
            ASSERT_GE(transferred, 0)
                << "GC must transfer the creator-retained handle ["
                << frame << "]";
            chain.grandchild_pidfd = transferred;
            have_grandchild = true;
        }
        else
        {
            if (transferred >= 0) ::close(transferred);
            FAIL() << "unexpected ack channel content [" << frame << "]";
        }
    }
    ASSERT_EQ(ack_rank, 0)
        << "the ack rank must be this single-rank launch's rank 0";

    // Healthy-liveness control: the transferred handle reports NO exit
    // notification at readiness.
    pollfd alive[] = {{chain.grandchild_pidfd, POLLIN, 0}};
    ASSERT_EQ(::poll(alive, 1, 0), 0)
        << "the sleeper must be alive at readiness";

    // AUTHENTICATED live session-root observation: this chain's isolated
    // session dir must host at least one REAL guardian root right now,
    // captured WITH stat identity (device+inode); iterator/stat errors
    // and empty observations both FAIL (a later empty check can never
    // pass vacuously).
    const auto live_roots = observeLiveSessionRoots(session_dir);
    ASSERT_TRUE(live_roots.has_value())
        << "session-root observation failed (iterator/stat error)";
    ASSERT_FALSE(live_roots->empty())
        << "no guardian root was observed in the chain-exclusive session "
           "dir during live readiness";

    // Cancellation AUTHENTICATION: a non-blocking exact-identity peek
    // (si_pid must be 0: a live authenticated child of this test).
    siginfo_t authenticated {};
    ASSERT_EQ(::waitid(P_PIDFD, chain.frontend_pidfd, &authenticated,
        WNOWAIT | WNOHANG | WEXITED), 0)
        << "pidfd authentication failed errno=" << errno;
    ASSERT_EQ(authenticated.si_pid, 0)
        << "the frontend must be a live authenticated child at cancellation";

    // THE CANCELLATION: SIGTERM through the authenticated retained pidfd.
    ASSERT_EQ(::syscall(SYS_pidfd_send_signal, chain.frontend_pidfd,
        SIGTERM, nullptr, 0), 0) << "cancel errno=" << errno;

    // The bounded EINTR-safe completion: the collection is recorded as
    // owned in-chain evidence (errors other than EINTR are fatal).
    int status = 0;
    const auto completion_deadline = std::chrono::steady_clock::now() +
        std::chrono::seconds(60);
    for (;;)
    {
        const pid_t waited = ::waitpid(chain.frontend, &status, WNOHANG);
        if (waited == chain.frontend)
        {
            chain.recordOwnedCollection(status);
            break;
        }
        if (waited < 0)
        {
            ASSERT_EQ(errno, EINTR) << "waitpid errno=" << errno;
            continue;
        }
        ASSERT_EQ(waited, 0) << "waitpid errno=" << errno;
        ASSERT_LT(std::chrono::steady_clock::now(), completion_deadline)
            << "the owned chain did not retire within the deadline";
        ::usleep(50000);
    }

    // The EXACT propagated cancellation status (owned in-chain evidence):
    // SIGTERM -> Signal packet -> guardian signalChildren -> mpirun
    // forwards to the job and exits 1 -> the guardian propagates
    // statusCode(mpirun)=1 -> the frontend's join returns 1.
    EXPECT_EQ(rawExit(chain.exit_status), 1)
        << "the frontend must propagate the guardian's cancellation status";

    // Surviving-parent ADOPTION evidence: the sleeper observed its own
    // reparenting and reported it (bounded, correlated to the GC record).
    const auto adopted_deadline = std::chrono::steady_clock::now() +
        std::chrono::seconds(30);
    bool have_adopted = false;
    long adopted_pid = -1;
    while (!have_adopted)
    {
        std::string frame;
        int transferred = -1;
        if (!receiveAck(chain.ack_socket, frame, transferred, adopted_deadline))
            FAIL() << "the sleeper never reported adoption "
                   "(surviving-parent evidence missing)";
        if (matchAckFrame(frame, "ADOPTED", ack.attempt, 2000000000L,
            adopted_pid))
        {
            ASSERT_EQ(transferred, -1)
                << "ADOPTED must not carry a handle [" << frame << "]";
            have_adopted = true;
        }
        else
        {
            if (transferred >= 0) ::close(transferred);
            FAIL() << "unexpected ack channel content [" << frame << "]";
        }
    }
    ASSERT_EQ(adopted_pid, reported_pid)
        << "the adoption record must name the same sleeper";

    // Grandchild death on the SAME transferred creator-retained handle.
    const auto death_deadline = std::chrono::steady_clock::now() +
        std::chrono::seconds(30);
    for (;;)
    {
        pollfd dead[] = {{chain.grandchild_pidfd, POLLIN, 0}};
        const int observed = ::poll(dead, 1, 200);
        if (observed == 1 && (dead[0].revents & POLLIN)) break;
        if (observed < 0)
        {
            ASSERT_EQ(errno, EINTR) << "death poll errno=" << errno;
            continue;
        }
        ASSERT_LT(std::chrono::steady_clock::now(), death_deadline)
            << "the adopted sleeper was not terminated within the bound";
    }

    // AUTHENTICATED session-root retirement: every captured stat identity
    // must be truly retired (no guardian-named entry may remain; the
    // validation never erases evidence — removal is teardown's separate
    // gated step).
    std::string retirement_failure;
    ASSERT_TRUE(validateRootRetirement(*live_roots, session_dir,
        retirement_failure)) << retirement_failure;

    // The explicit bounded teardown: a valid OWNED completion with no
    // notes; the session gate re-proves retirement before removal.
    const auto result = chain.teardown();
    ASSERT_EQ(result.status, TeardownStatus::Completed) << chain.report;
    ASSERT_TRUE(result.reaped);
    ASSERT_TRUE(result.has_exit_status);
    ASSERT_TRUE(result.report.empty()) << result.report;
    ASSERT_FALSE(std::filesystem::exists(chain.owner_root));
    ASSERT_FALSE(std::filesystem::exists(session_dir));
}

TEST(PlacementIntegration, ReadinessTimeoutRetiresOwnedChainBounded)
{
    // CONTROL: the readiness deadline's FAILURE path must retire the whole
    // owned production chain bounded (a receiverless ack path; no RDY ever
    // arrives; the explicit teardown must reap the live frontend with the
    // pinned propagated status, prove the authenticated live session
    // roots retired through its gated scan, and remove both test-owned
    // directories — reported, never hung, never abandoned, never
    // vacuous, never deleting live guardian roots).
    const auto cpus = admittedCpus();
    ASSERT_GE(cpus.size(), 2u);
    ScopedEnvironment env;
    env.unset("OMP_NUM_THREADS");
    env.unset("OMP_PLACES");
    env.unset("OMP_PROC_BIND");
    env.set("LLAMINAR_MANAGED_PLACEMENT", "1");
    env.set("LLAMINAR_NATIVE_CHILD_PARK", "1");

    CancellationChain chain;
    AckSetup ack;
    ASSERT_TRUE(ack.create("timeout", true))
        << "ack setup failed errno=" << errno;
    adoptAckSetup(chain, ack);
    const std::string session_dir = chain.session_dir;
    env.set("TMPDIR", session_dir);
    // The rank's ack socket points at a receiverless path: its connect
    // fails and the RDY never arrives (the controlled timeout condition).
    env.set("LLAMINAR_NATIVE_CHILD_ACK_SOCK",
        chain.owner_root + "/never-bound.sock");
    env.set("LLAMINAR_NATIVE_CHILD_ATTEMPT", ack.attempt);

    const int phase = spawnFrontendWithExactHandle(chain);
    if (phase == SpawnChildExec) execProductionFrontend(cpus, ack.attempt);
    ASSERT_EQ(phase, SpawnedWithHandle);

    const auto short_deadline = std::chrono::steady_clock::now() +
        std::chrono::seconds(5);
    std::string frame;
    int transferred = -1;
    const bool arrived = receiveAck(chain.ack_socket, frame, transferred,
        short_deadline);
    if (transferred >= 0) ::close(transferred);
    ASSERT_FALSE(arrived)
        << "no ack may arrive on a receiverless path [" << frame << "]";

    // AUTHENTICATED live roots BEFORE teardown (the chain is live even
    // without RDY: mpirun and the guardian are running), captured with
    // stat identity.
    const auto live_roots = observeLiveSessionRoots(session_dir);
    ASSERT_TRUE(live_roots.has_value())
        << "session-root observation failed (iterator/stat error)";
    ASSERT_FALSE(live_roots->empty())
        << "no guardian root observed while the chain was live";

    const auto teardown_started = std::chrono::steady_clock::now();
    const auto result = chain.teardown();
    ASSERT_LT(std::chrono::steady_clock::now() - teardown_started,
        std::chrono::seconds(60)) << "teardown must stay bounded";
    ASSERT_EQ(result.status, TeardownStatus::Completed) << chain.report;
    ASSERT_TRUE(result.reaped)
        << "the teardown must have exactly collected the frontend";
    ASSERT_TRUE(result.has_exit_status)
        << "the teardown owns the collected native status";
    EXPECT_EQ(rawExit(result.exit_status), 1)
        << "the teardown-terminated chain must propagate status 1";
    ASSERT_TRUE(recordedIdentitiesGone(*live_roots))
        << "the authenticated live root identities were not retired";
    ASSERT_FALSE(std::filesystem::exists(chain.owner_root));
    ASSERT_FALSE(std::filesystem::exists(session_dir));
}

TEST(PlacementIntegration, TeardownHandlesLostWaitOwnership)
{
    // FAILURE INJECTION — lost wait ownership: an external collection of
    // the frontend (this test's own waitpid, NOT recorded in the chain)
    // makes teardown's waitpid observe ECHILD. Death was already certified
    // by the retained handle's POLLIN oracle; the outcome is the DISTINCT
    // CompletedCleanupOnly state — the native exit status was never
    // observed in-chain (has_exit_status false), the lost ownership is a
    // visible note, and the cleanup still completes bounded with the root
    // removed. It is never reported as a valid owned completion.
    const auto cpus = admittedCpus();
    ASSERT_GE(cpus.size(), 2u);
    ScopedEnvironment env;
    env.unset("OMP_NUM_THREADS");
    env.unset("OMP_PLACES");
    env.unset("OMP_PROC_BIND");
    env.set("LLAMINAR_MANAGED_PLACEMENT", "1");
    env.set("LLAMINAR_NATIVE_CHILD_PARK", "1");

    CancellationChain chain;
    AckSetup ack;
    ASSERT_TRUE(ack.create("lostwait", false))
        << "ack setup failed errno=" << errno;
    adoptAckSetup(chain, ack);
    env.set("LLAMINAR_NATIVE_CHILD_ACK_SOCK", chain.ack_socket_path);
    env.set("LLAMINAR_NATIVE_CHILD_ATTEMPT", ack.attempt);

    const int phase = spawnFrontendWithExactHandle(chain);
    if (phase == SpawnChildExec) execProductionFrontend(cpus, ack.attempt);
    ASSERT_EQ(phase, SpawnedWithHandle);
    ASSERT_TRUE(awaitRankReady(chain, ack, std::chrono::seconds(30)))
        << "readiness deadline exceeded";

    // THE INJECTION: cancel through the exact handle, then collect the
    // frontend EXTERNALLY (this test's own waitpid) WITHOUT recording it
    // in the chain — teardown must then lose wait ownership.
    ASSERT_EQ(::syscall(SYS_pidfd_send_signal, chain.frontend_pidfd,
        SIGTERM, nullptr, 0), 0);
    int status = 0;
    const auto completion_deadline = std::chrono::steady_clock::now() +
        std::chrono::seconds(60);
    for (;;)
    {
        const pid_t waited = ::waitpid(chain.frontend, &status, WNOHANG);
        if (waited == chain.frontend) break;
        if (waited < 0)
        {
            ASSERT_EQ(errno, EINTR);
            continue;
        }
        ASSERT_EQ(waited, 0);
        ASSERT_LT(std::chrono::steady_clock::now(), completion_deadline);
        ::usleep(50000);
    }
    // chain.reaped stays FALSE: teardown must observe the lost ownership.

    const auto result = chain.teardown();
    ASSERT_EQ(result.status, TeardownStatus::CompletedCleanupOnly)
        << "lost wait ownership must be the distinct cleanup-only state: "
        << chain.report;
    ASSERT_TRUE(result.collected_elsewhere)
        << "teardown must have detected the lost wait ownership";
    ASSERT_NE(result.report.find("wait ownership lost"), std::string::npos)
        << "the lost-ownership state must be visible: " << chain.report;
    ASSERT_FALSE(result.reaped)
        << "only an exact in-chain waitpid collection may set reaped";
    ASSERT_FALSE(result.has_exit_status)
        << "the native status was collected elsewhere; the chain must not "
           "claim owned status evidence";
    ASSERT_FALSE(std::filesystem::exists(chain.owner_root));
}

TEST(PlacementIntegration, TeardownRetriesThroughEintr)
{
    // FAILURE INJECTION — EINTR storm: a bounded child storms the test
    // process with SIGUSR1 (a non-terminating diagnostic signal, delivered
    // through the child's pidfd handle of its parent — no numeric-pid
    // signaling) while teardown runs. The handler is installed WITHOUT
    // SA_RESTART, so poll/usleep/waitpid paths observe EINTR; teardown
    // must retry within its bounds and complete a clean OWNED retirement.
    const auto cpus = admittedCpus();
    ASSERT_GE(cpus.size(), 2u);
    ScopedEnvironment env;
    env.unset("OMP_NUM_THREADS");
    env.unset("OMP_PLACES");
    env.unset("OMP_PROC_BIND");
    env.set("LLAMINAR_MANAGED_PLACEMENT", "1");
    env.set("LLAMINAR_NATIVE_CHILD_PARK", "1");

    CancellationChain chain;
    AckSetup ack;
    ASSERT_TRUE(ack.create("eintr", false))
        << "ack setup failed errno=" << errno;
    adoptAckSetup(chain, ack);
    env.set("LLAMINAR_NATIVE_CHILD_ACK_SOCK", chain.ack_socket_path);
    env.set("LLAMINAR_NATIVE_CHILD_ATTEMPT", ack.attempt);

    const int phase = spawnFrontendWithExactHandle(chain);
    if (phase == SpawnChildExec) execProductionFrontend(cpus, ack.attempt);
    ASSERT_EQ(phase, SpawnedWithHandle);
    ASSERT_TRUE(awaitRankReady(chain, ack, std::chrono::seconds(30)))
        << "readiness deadline exceeded";

    // The SIGUSR1 storm: a no-restart handler plus a bounded storm child.
    struct sigaction action {};
    action.sa_handler = [](int) {};
    ASSERT_EQ(::sigaction(SIGUSR1, &action, nullptr), 0);
    const pid_t storm = ::fork();
    ASSERT_GE(storm, 0);
    if (storm == 0)
    {
        // Storm the parent through its exact handle (pidfd_open of the
        // parent; pidfd_send_signal of SIGUSR1 — a diagnostic signal, not
        // a termination; bounded to 2000 sends then exit 0).
        const int parent_handle = static_cast<int>(
            ::syscall(SYS_pidfd_open, ::getppid(), 0));
        if (parent_handle >= 0)
            for (int i = 0; i < 2000; ++i)
            {
                ::syscall(SYS_pidfd_send_signal, parent_handle, SIGUSR1,
                    nullptr, 0);
                ::usleep(1000);
            }
        std::_Exit(0);
    }

    // Cancel and tear down UNDER the storm: every EINTR is retried inside
    // the bounds and the retirement completes as an owned completion.
    ASSERT_EQ(::syscall(SYS_pidfd_send_signal, chain.frontend_pidfd,
        SIGTERM, nullptr, 0), 0);
    const auto result = chain.teardown();
    ASSERT_EQ(result.status, TeardownStatus::Completed) << chain.report;
    ASSERT_TRUE(result.reaped);
    ASSERT_TRUE(result.has_exit_status);
    ASSERT_FALSE(std::filesystem::exists(chain.owner_root));

    // Collect the bounded storm child (EINTR-safe).
    const auto storm_deadline = std::chrono::steady_clock::now() +
        std::chrono::seconds(10);
    for (;;)
    {
        int storm_status = 0;
        const pid_t waited = ::waitpid(storm, &storm_status, WNOHANG);
        if (waited == storm) break;
        if (waited < 0)
        {
            ASSERT_EQ(errno, EINTR);
            continue;
        }
        ASSERT_EQ(waited, 0);
        ASSERT_LT(std::chrono::steady_clock::now(), storm_deadline);
        ::usleep(20000);
    }
}

TEST(PlacementIntegration, TeardownAcquisitionFailureIsFatalThenReacquireRetires)
{
    // FAILURE INJECTION — REAL exact-handle acquisition failure: the
    // parent's soft RLIMIT_NOFILE is lowered to the next allocatable fd
    // number, so the immediate post-fork pidfd_open fails EMFILE (a
    // genuine acquisition failure, not a simulated flag). The structured
    // spawn still leaves the LIVE child owned by the chain. While the fd
    // budget stays exhausted, teardown's re-acquisition also fails: a
    // VISIBLE FATAL with the root AND session backing RETAINED and the
    // chain left owned (no numeric-pid fallback anywhere). After the
    // budget is restored, a supervising teardown re-acquires the exact
    // handle through the same primitive and completes a valid owned
    // retirement. The scoped exhaustion owns restoration on EVERY path.
    const auto cpus = admittedCpus();
    ASSERT_GE(cpus.size(), 2u);
    ScopedEnvironment env;
    env.unset("OMP_NUM_THREADS");
    env.unset("OMP_PLACES");
    env.unset("OMP_PROC_BIND");
    env.set("LLAMINAR_MANAGED_PLACEMENT", "1");
    env.set("LLAMINAR_NATIVE_CHILD_PARK", "1");

    CancellationChain chain;
    AckSetup ack;
    ASSERT_TRUE(ack.create("acqfail", false))
        << "ack setup failed errno=" << errno;
    adoptAckSetup(chain, ack);
    env.set("LLAMINAR_NATIVE_CHILD_ACK_SOCK", chain.ack_socket_path);
    env.set("LLAMINAR_NATIVE_CHILD_ATTEMPT", ack.attempt);

    // Declared AFTER the chain/ack owners: its restoration runs BEFORE
    // their cleanup on every path (reverse-declaration destruction).
    ScopedNoFileExhaustion fd_budget;
    ASSERT_TRUE(fd_budget.arm()) << "could not arm the fd exhaustion";
    const int acq_errno = spawnFrontendWithForcedAcquisitionFailure(chain,
        fd_budget, [&] { execProductionFrontend(cpus, ack.attempt); });
    ASSERT_EQ(acq_errno, EMFILE)
        << "the exact-handle acquisition must genuinely fail EMFILE, got "
        << acq_errno;
    ASSERT_GT(chain.frontend, 0)
        << "the live child must be owned by the chain despite the failed "
           "acquisition";
    ASSERT_EQ(chain.frontend_pidfd, -1);

    // The chain went LIVE despite the failed acquisition (liveness does
    // not depend on the handle; the ack socket was bound before the fd
    // budget was lowered and poll allocates nothing).
    ASSERT_TRUE(awaitRankReady(chain, ack, std::chrono::seconds(30)))
        << "readiness deadline exceeded";

    // First teardown UNDER the exhausted fd budget: the re-acquisition
    // fails EMFILE again -> a VISIBLE FATAL; the backing is RETAINED;
    // the live chain stays owned and RETRYABLE; nothing is terminated
    // by fallback.
    const auto failed = chain.teardown();
    ASSERT_EQ(failed.status, TeardownStatus::Failed) << chain.report;
    ASSERT_NE(failed.report.find(
        "FATAL: exact frontend handle could not be (re)acquired"),
        std::string::npos) << chain.report;
    ASSERT_NE(failed.report.find("RETAINED for evidence"), std::string::npos)
        << chain.report;
    ASSERT_FALSE(failed.reaped);
    ASSERT_TRUE(std::filesystem::exists(chain.owner_root))
        << "the ack root must be retained on the failed retirement";

    // Restore the fd budget; the supervising teardown re-acquires the
    // exact handle through the SAME primitive on the still-owned
    // unreaped child and completes the retirement bounded.
    fd_budget.restore_now();
    const auto recovered = chain.teardown();
    ASSERT_EQ(recovered.status, TeardownStatus::Completed) << chain.report;
    ASSERT_TRUE(recovered.reaped);
    ASSERT_TRUE(recovered.has_exit_status);
    EXPECT_EQ(rawExit(recovered.exit_status), 1)
        << "the recovered teardown must retire the live production chain";
    ASSERT_FALSE(std::filesystem::exists(chain.owner_root));
}

namespace {

/// Static state bridge for the EXPECT_FATAL_FAILURE checked harness: its
/// statement executes inside a static member function and may not
/// reference local non-static variables, so the harness body reaches the
/// enclosing test's state through this file-local pointer.
struct FatalHarnessState
{
    std::vector<int> cpus;
    ScopedEnvironment *env = nullptr;
    CancellationChain::DestructorRetryEvidence *evidence = nullptr;
    bool reached_after_failure = false;
    std::string owner_path, session_path;
};
FatalHarnessState *g_fatal_harness = nullptr;

/// The checked fatal-assert harness body (review finding): owns the
/// chain, ack setup and fd exhaustion as ITS locals. The FATAL readiness
/// ASSERT fires on the receiverless timeout and returns early FROM THIS
/// FUNCTION — code after it must not execute — and the RAII locals then
/// run in reverse-declaration order: the fd budget is restored (and
/// verified) BEFORE the chain's supervised destructor retry re-acquires
/// the exact handle and retires the live chain. EXPECT_FATAL_FAILURE
/// intercepts the expected fatal failure (the enclosing test stays
/// green; any OTHER failure, including destructor ADD_FAILUREs, breaks
/// the exactly-one-fatal contract and fails the test).
void readinessFatalHarness()
{
    FatalHarnessState &state = *g_fatal_harness;
    CancellationChain chain;
    AckSetup ack;
    ASSERT_TRUE(ack.create("rdfail", true))
        << "ack setup failed errno=" << errno;
    adoptAckSetup(chain, ack);
    state.owner_path = chain.owner_root;
    state.session_path = chain.session_dir;
    state.env->set("TMPDIR", state.session_path);
    state.env->set("LLAMINAR_NATIVE_CHILD_ACK_SOCK",
        state.owner_path + "/never-bound.sock");
    state.env->set("LLAMINAR_NATIVE_CHILD_ATTEMPT", ack.attempt);

    // Declared AFTER the chain/ack owners: its restoration runs BEFORE
    // their cleanup when this function returns (fatal early return
    // included).
    ScopedNoFileExhaustion fd_budget;
    ASSERT_TRUE(fd_budget.arm()) << "could not arm the fd exhaustion";
    const int acq_errno = spawnFrontendWithForcedAcquisitionFailure(
        chain, fd_budget,
        [&] { execProductionFrontend(state.cpus, ack.attempt); });
    ASSERT_EQ(acq_errno, EMFILE)
        << "the exact-handle acquisition must genuinely fail EMFILE, got "
        << acq_errno;
    ASSERT_GT(chain.frontend, 0);
    ASSERT_EQ(chain.frontend_pidfd, -1);
    chain.destructor_evidence = state.evidence;

    // THE INJECTED FATAL READINESS FAILURE: the receiverless path times
    // out, this ASSERT FIRES, and the function returns early —
    // everything below must NOT run.
    ASSERT_TRUE(awaitRankReady(chain, ack, std::chrono::seconds(5)))
        << "readiness failed (receiverless path): fatal early return";
    state.reached_after_failure = true; // MUST NOT EXECUTE
    FAIL() << "unreachable after a fired fatal assert";
}

} // namespace

TEST(PlacementIntegration, ReadinessFailureUnderExhaustedFdsRestoresAndRetires)
{
    // FAILURE INJECTION — readiness failure while the fd budget is
    // exhausted, through the CHECKED FATAL-ASSERT HARNESS the reviewer
    // prescribed: EXPECT_FATAL_FAILURE runs readinessFatalHarness() and
    // verifies EXACTLY ONE fatal failure with the expected message fires
    // (the real ASSERT early-return semantics — not a passing assert and
    // normal scope exit). The harness's RAII scopes run in order at the
    // early return. EVERY recovery outcome is then asserted OUTSIDE the
    // harness: the must-not-execute marker stayed false, the original
    // RLIMIT_NOFILE is restored, the chain's supervised destructor retry
    // ran and collected the exact frontend with the pinned propagated
    // cancellation status, and both owned directories retired.
    const auto cpus = admittedCpus();
    ASSERT_GE(cpus.size(), 2u);
    ScopedEnvironment env;
    env.unset("OMP_NUM_THREADS");
    env.unset("OMP_PLACES");
    env.unset("OMP_PROC_BIND");
    env.set("LLAMINAR_MANAGED_PLACEMENT", "1");
    env.set("LLAMINAR_NATIVE_CHILD_PARK", "1");

    rlimit original {};
    ASSERT_EQ(::getrlimit(RLIMIT_NOFILE, &original), 0);
    CancellationChain::DestructorRetryEvidence evidence {};
    FatalHarnessState state;
    state.cpus = cpus;
    state.env = &env;
    state.evidence = &evidence;
    g_fatal_harness = &state;

    // The checked harness: the expected fatal fires INSIDE; the test
    // continues here after the harness function's early return and RAII
    // unwinding.
    EXPECT_FATAL_FAILURE(readinessFatalHarness(),
        "readiness failed (receiverless path)");
    g_fatal_harness = nullptr;

    // OUTSIDE the harness — the early return happened and the RAII
    // scopes ran in order. The must-not-execute marker stayed false.
    EXPECT_FALSE(state.reached_after_failure)
        << "the fatal assert must have returned early from the harness";
    // The fd budget was restored to the ORIGINAL limit (verified inside
    // the scoped destructor AND re-checked here from outside).
    rlimit now {};
    ASSERT_EQ(::getrlimit(RLIMIT_NOFILE, &now), 0);
    ASSERT_EQ(now.rlim_cur, original.rlim_cur)
        << "the original RLIMIT_NOFILE must be restored after the fatal "
           "path";
    // The chain's supervised destructor retry RAN and completed an OWNED
    // retirement: exact frontend collected, pinned propagated status 1.
    ASSERT_TRUE(evidence.ran)
        << "the chain destructor must have performed the supervised retry";
    ASSERT_EQ(evidence.result.status, TeardownStatus::Completed)
        << evidence.result.report;
    ASSERT_TRUE(evidence.result.reaped)
        << "the exact frontend must be collected after the fatal path";
    ASSERT_TRUE(evidence.result.has_exit_status);
    EXPECT_EQ(rawExit(evidence.result.exit_status), 1)
        << "the recovered retirement must propagate the cancellation "
           "status";
    // Both owned directories retired (empty rmdir; no residue).
    ASSERT_FALSE(std::filesystem::exists(state.owner_path))
        << "the ack root must be retired after the fatal path";
    ASSERT_FALSE(std::filesystem::exists(state.session_path))
        << "the session backing must be retired after the fatal path";
}

TEST(PlacementIntegration, TeardownEscalatesToSigkillAfterTermBound)
{
    // FAILURE INJECTION — SIGTERM bound exceeded: a rank that IGNORES
    // SIGTERM (the park-mode ignore variant) keeps the chain alive past
    // teardown's (shortened) SIGTERM bound; teardown must escalate to a
    // CHECKED SIGKILL on the same exact handle, complete the collection
    // bounded, and record the escalation visibly. (The guardian
    // independently retires its own subtree with its own SIGKILL grace;
    // this test asserts the TEST-side escalation only.)
    ScopedEnvironment env;
    // The injection target is the native-child helper itself in
    // ignore+park mode, forked DIRECTLY with the same exact-handle
    // acquisition: a live authenticated child that ignores SIGTERM. (The
    // production mpirun chain cannot honestly produce this condition —
    // mpirun retires behind an ignoring rank — and its retirement is
    // already proven by the main cancellation test.)
    env.set("LLAMINAR_NATIVE_CHILD_PARK", "1");
    env.set("LLAMINAR_NATIVE_CHILD_IGNORE_SIGTERM", "1");
    // The ack contract names the launched rank (OMPI_COMM_WORLD_RANK); the
    // direct exec supplies the rank-0 context explicitly.
    env.set("OMPI_COMM_WORLD_RANK", "0");

    CancellationChain chain;
    AckSetup ack;
    ASSERT_TRUE(ack.create("sigkill", false))
        << "ack setup failed errno=" << errno;
    adoptAckSetup(chain, ack);
    env.set("LLAMINAR_NATIVE_CHILD_ACK_SOCK", chain.ack_socket_path);
    env.set("LLAMINAR_NATIVE_CHILD_ATTEMPT", ack.attempt);
    // Shortened bounds so the escalation path runs within the gate.
    chain.term_bound = std::chrono::seconds(2);
    chain.kill_bound = std::chrono::seconds(5);

    const int phase = spawnFrontendWithExactHandle(chain);
    if (phase == SpawnChildExec)
    {
        char *argv[] = {const_cast<char *>(LLAMINAR_PLACEMENT_NATIVE_CHILD_PATH),
            nullptr};
        ::execv(argv[0], argv);
        std::_Exit(127);
    }
    ASSERT_EQ(phase, SpawnedWithHandle);
    ASSERT_TRUE(awaitRankReady(chain, ack, std::chrono::seconds(30)))
        << "readiness deadline exceeded";

    // THE CANCELLATION + the shortened-bound teardown: the SIGTERM to the
    // child is ignored; the checked SIGKILL escalation retires it.
    ASSERT_EQ(::syscall(SYS_pidfd_send_signal, chain.frontend_pidfd,
        SIGTERM, nullptr, 0), 0);
    const auto result = chain.teardown();
    ASSERT_EQ(result.status, TeardownStatus::Completed) << chain.report;
    ASSERT_TRUE(result.reaped);
    ASSERT_TRUE(result.has_exit_status);
    ASSERT_NE(result.report.find("escalated to SIGKILL"), std::string::npos)
        << "the escalation must be visible: " << chain.report;
    ASSERT_FALSE(std::filesystem::exists(chain.owner_root));
}
