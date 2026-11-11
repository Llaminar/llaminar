/**
 * @file Test__MPIBootstrap.cpp
 * @brief Device-free launch policy and typed NUMA intent regressions.
 *
 * Test command construction with synthetic topology and validate malformed
 * placement before any hardware discovery, MPI launch, or model allocation.
 * Actual worker affinity belongs in the model-free CPU startup integration.
 * Malformed native place receipts fail before exec or worker admission; the
 * opaque native place token cannot be default-constructed or copied.
 * Diagnostic export tests deliberately distinguish portable observation policy
 * from launcher-local device visibility and cloud authentication environment.
 */

#include "utils/MPIBootstrap.h"
#include "utils/DebugEnv.h"
#include "utils/MPISessionOwner.h"
#include "fixtures/cpu/RecordedGB10NativeCorePlaces.h"
#include "utils/OpenMPUtils.h"
#include "utils/PerfStatsCollector.h"
#include "app/MPIBootstrapPhase.h"
#include "backends/ComputeBackend.h"
#include "backends/HardwareInventory.h"
#include "config/ExecutionDomainDefinition.h"
#include "config/OrchestrationConfigParser.h"
#include "config/OrchestrationConfigDocument.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <unistd.h>
#include <optional>
#include <thread>
#include <chrono>
#include <memory>
#include <cstdint>
#include <iostream>
#include <string>
#include <vector>
#include <array>
#include "utils/NUMATopology.h"
#include <type_traits>
#include <thread>

namespace
{
    /** @brief Restore each mutated environment variable on every test exit. */
    class ScopedLaunchEnvironment final
    {
    public:
        /** @brief Set an explicit value, or remove it to exercise absent intent. */
        ScopedLaunchEnvironment(const char *name, const char *value) : name_(name)
        {
            if (const char *previous = std::getenv(name))
                previous_ = previous;
            if (value)
                ::setenv(name, value, 1);
            else
                ::unsetenv(name);
        }

        /** @brief Restore absence as absence, including assertion exits. */
        ~ScopedLaunchEnvironment()
        {
            if (previous_)
                ::setenv(name_.c_str(), previous_->c_str(), 1);
            else
                ::unsetenv(name_.c_str());
        }

        ScopedLaunchEnvironment(const ScopedLaunchEnvironment &) = delete;
        ScopedLaunchEnvironment &operator=(const ScopedLaunchEnvironment &) = delete;

    private:
        std::string name_;
        std::optional<std::string> previous_;
    };

    /** @return MPI command for a synthetic topology without starting MPI. */
    std::vector<std::string> buildCommand(llaminar2::MPILaunchConfig config,
                                          llaminar2::CPUTopology topology)
    {
        char arg0[] = "llaminar2";
        char arg1[] = "serve";
        char *argv[] = {arg0, arg1};
        return llaminar2::MPIBootstrap::buildMPIRunCommand(
            2,
            argv,
            config,
            topology,
            llaminar2::MPIBootstrapArgumentPolicy::FilterLaunchArguments,
            config.numa_mapping ? "/fixture/original-observed-numa.rankfile" : "");
    }

    /** @return Whether the constructed command contains an exact argument. */
    bool contains(const std::vector<std::string> &values,
                  const std::string &needle)
    {
        return std::find(values.begin(), values.end(), needle) != values.end();
    }

} // namespace

/**
 * @test Bootstrap leaves native transport selection and explicit operator intent intact.
 * @details MPI owns available local-copy mechanisms. Command-line overrides take
 * precedence over its environment, so a forced disabled mechanism would defeat
 * both native hardware selection and an explicit operator CMA setting.
 */
TEST(Test__MPIBootstrap, NativeTransportSelectionPreservesAbsentAndExplicitIntent)
{
    constexpr const char *name="OMPI_MCA_btl_vader_single_copy_mechanism";
    for(const char *hostfile:{"","/cluster/hosts"})
    for(const int ranks:{1,2,4,8})
    for(const char *intent:{static_cast<const char *>(nullptr),"none","cma"})
    {
        SCOPED_TRACE(::testing::Message()<<hostfile<<'/'<<ranks<<'/'<<(intent?intent:"native default"));
        ScopedLaunchEnvironment environment(name,intent);
        llaminar2::MPILaunchConfig config;
        config.hostfile=hostfile;
        config.num_procs=ranks;
        const auto command=buildCommand(config,{});
        EXPECT_FALSE(contains(command,"btl_vader_single_copy_mechanism"));
        const char *remaining=std::getenv(name);
        if(intent) { ASSERT_NE(remaining,nullptr); EXPECT_STREQ(remaining,intent); }
        else EXPECT_EQ(remaining,nullptr);
    }
}

/** Remote SSH ranks receive observation intent through MPI, not local inheritance. */
TEST(Test__MPIBootstrap, PortableDiagnosticsAreExplicitNameOnlyExports)
{
    const char *names[] = {
        "LLAMINAR_LOG_LEVEL", "LLAMINAR_PERF_STATS_JSON",
        "LLAMINAR_PERF_STATS_CSV", "LLAMINAR_PERF_STATS_FILTER",
        "LLAMINAR_PERF_STATS_SUMMARY", "LLAMINAR_PERF_STATS_TABLE",
        "LLAMINAR_PERF_STATS_TABLE_LIMIT", "LLAMINAR_PERF_STATS_CPU_STAGE_TIMING",
        "LLAMINAR_PERF_STATS_GPU_STAGE_TIMING", "LLAMINAR_GPU_STAGE_TIMING",
        "LLAMINAR_GPU_STAGE_TIMING_DETAIL", "LLAMINAR_PROFILING"};
    for (const char *hostfile : {"", "/cluster/hosts"})
    {
        llaminar2::MPILaunchConfig config;
        config.hostfile = hostfile;
        for (const char *name : names)
        {
            for (const char *value : {"/output with spaces/rank-{rank}.json", "0", ""})
            {
                ScopedLaunchEnvironment environment(name, value);
                const auto command = buildCommand(config, {});
                const auto found = std::find(command.begin(), command.end(), name);
                ASSERT_NE(found, command.end()) << name;
                ASSERT_NE(found, command.begin());
                EXPECT_EQ(*(found - 1), "-x");
                EXPECT_LT(found, std::find(command.begin(), command.end(), "llaminar2"));
                EXPECT_EQ(std::count(command.begin(), command.end(), name), 1);
                EXPECT_FALSE(contains(command, value));
            }
            ScopedLaunchEnvironment absent(name, nullptr);
            EXPECT_FALSE(contains(buildCommand(config, {}), name));
        }
    }
}

/** The bounded libgomp idle-spin budget is installed once and reaches every rank by name. */
TEST(Test__MPIBootstrap, OpenMPIdleSpinPolicyIsBoundedAndExportedByName)
{
    {
        ScopedLaunchEnvironment absent("GOMP_SPINCOUNT", nullptr);
        llaminar2::MPIBootstrap::configureOpenMPWaitPolicy();
        ASSERT_NE(std::getenv("GOMP_SPINCOUNT"), nullptr);
        EXPECT_STREQ(std::getenv("GOMP_SPINCOUNT"), llaminar2::MPIBootstrap::kOpenMPIdleSpinCount);
        // Bounded: never an unbounded active wait that keeps an idle server busy.
        EXPECT_STRNE(std::getenv("GOMP_SPINCOUNT"), "infinite");
        for (const char *hostfile : {"", "/cluster/hosts"})
        {
            llaminar2::MPILaunchConfig config;
            config.hostfile = hostfile;
            const auto command = buildCommand(config, {});
            const auto found = std::find(command.begin(), command.end(), "GOMP_SPINCOUNT");
            ASSERT_NE(found, command.end());
            EXPECT_EQ(*(found - 1), "-x");
            EXPECT_FALSE(contains(command, llaminar2::MPIBootstrap::kOpenMPIdleSpinCount));
        }
    }
    {
        // An explicit measurement value is preserved, never overwritten.
        ScopedLaunchEnvironment explicit_value("GOMP_SPINCOUNT", "30000000");
        llaminar2::MPIBootstrap::configureOpenMPWaitPolicy();
        EXPECT_STREQ(std::getenv("GOMP_SPINCOUNT"), "30000000");
    }
}

/**
 * @brief The OpenMP team-fork counter's publication and self-cost contract.
 *
 * Default serving materializes no "omp" record: the fork sites increment a
 * relaxed atomic unconditionally (the production truth survives every run),
 * while both publication families stay behind the collection policy, so an
 * off-default process carries the one-atomic cost and nothing else. Under an
 * evidence configuration the summary and stage families publish through the
 * same authority the engine and graph executors use, and the note's own
 * amortized per-invocation cost is measured and recorded: the wake-up work
 * item's ns-scale claim becomes a published receipt, not an assertion.
 */
TEST(Test__MPIBootstrap, OpenMPTeamForkCounterDefaultAndPublication)
{
    using namespace llaminar2;
    // Off-default: no collection policy selected anything.
    {
        const ScopedLaunchEnvironment no_filter("LLAMINAR_PERF_STATS_FILTER", nullptr);
        const ScopedLaunchEnvironment no_summary("LLAMINAR_PERF_STATS_SUMMARY", nullptr);
        const ScopedLaunchEnvironment no_table("LLAMINAR_PERF_STATS_TABLE", nullptr);
        const ScopedLaunchEnvironment no_json("LLAMINAR_PERF_STATS_JSON", nullptr);
        const ScopedLaunchEnvironment no_csv("LLAMINAR_PERF_STATS_CSV", nullptr);
        const ScopedLaunchEnvironment no_profiling("LLAMINAR_PROFILING", nullptr);
        const ScopedLaunchEnvironment no_cpu_timing("LLAMINAR_PERF_STATS_CPU_STAGE_TIMING", nullptr);
        const ScopedLaunchEnvironment no_gpu_timing("LLAMINAR_PERF_STATS_GPU_STAGE_TIMING", nullptr);
        mutableDebugEnv().reload();
        PerfStatsCollector::reloadConfigurationFromEnvironment();
        PerfStatsCollector::reset();
        EXPECT_FALSE(PerfStatsCollector::isRecordFamilyEnabled(
            "omp", "openmp_team_forks_decode_step"));
        EXPECT_FALSE(PerfStatsCollector::isRecordFamilyEnabled(
            "omp", "openmp_team_forks_stage"));
        // Forks still count in production truth while nothing materializes.
        const std::uint64_t before = llaminar::v2::openMPTeamForkSnapshot();
        auto work = [] {};
        OMP_WORKSHARE_REGION(work);
        OMP_WORKSHARE_REGION_SYNC(work);
        EXPECT_EQ(llaminar::v2::openMPTeamForkSnapshot() - before, 2u);
    }

    // Self-cost evidence, under the ownership configurations production runs:
    // no owner, the one owner a window/stage holds, the two nested owners, and
    // two concurrent owners. Each variant is amortized over a million notes,
    // recorded in the collector and logged, and bounded generously (the fork
    // it describes costs microseconds, so the observation must never
    // approach that scale).
    {
        // The evidence configuration lives in this inner scope so BOTH RAII
        // guards complete BEFORE the final reload below: a reload that ran
        // while the live guards could still mutate the environment would
        // leave the authorities describing a configuration nobody set.
        const ScopedLaunchEnvironment filter("LLAMINAR_PERF_STATS_FILTER", "omp");
        const ScopedLaunchEnvironment summary("LLAMINAR_PERF_STATS_SUMMARY", "1");
        mutableDebugEnv().reload();
        PerfStatsCollector::reloadConfigurationFromEnvironment();
        EXPECT_TRUE(PerfStatsCollector::isRecordFamilyEnabled(
            "omp", "openmp_team_forks_decode_step"));
        EXPECT_TRUE(PerfStatsCollector::isRecordFamilyEnabled(
            "omp", "openmp_team_forks_stage"));

        constexpr std::uint64_t kMeasuredNotes = 1u << 20;
        // Per-owner note latency, measured under the ownership configurations
        // production runs. The body matches what the basis label claims:
        // owners==0 constructs NO scope; the scope configuration is built OFF
        // the clock; the concurrent worker exists and has signalled readiness
        // with its own same-thread scope before the start gate opens (the
        // measuring main thread owns a cell too, so contention is between two
        // real owners); the timed interval is the amortized concurrent wall
        // time INCLUDING the rendezvous (start-gate publication through the
        // competing owner's observed completion - readiness permits
        // concurrency and does not prove scheduler overlap); cell and
        // process verification, the join and the teardown all happen after
        // the clock stops; the LIFO teardown is enforced inner-first with
        // the exact owner-stack pointer identity asserted after it. The
        // concurrent figure divides by the notes ONE thread executed:
        // per-owner latency, not aggregate throughput. Earlier no-walk
        // figures do not transfer and are superseded.
        const auto measure = [&](int owners, bool concurrent) -> double
        {
            const std::uint64_t before = llaminar::v2::openMPTeamForkSnapshot();
            // The ORIGINAL owner-stack pointer, saved before any scope
            // exists: restoration is proven by POINTER IDENTITY against it,
            // never by global-total arithmetic.
            const llaminar::v2::OpenMPTeamForkCell *stack_at_entry =
                llaminar::v2::openMPTeamForkOwnerStack;
            std::optional<llaminar::v2::OpenMPTeamForkScope> outer;
            std::optional<llaminar::v2::OpenMPTeamForkScope> inner;
            std::optional<llaminar::v2::OpenMPTeamForkScope> main_owner;
            std::atomic<bool> worker_ready{false};
            std::atomic<bool> start_gate{false};
            std::atomic<bool> worker_notes_done{false};
            std::atomic<bool> verify_gate{false};
            std::thread worker;
            if (concurrent)
            {
                main_owner.emplace(); // The measuring thread owns a cell too.
                worker = std::thread([&]
                {
                    // The worker asserts its OWN stack identity around its
                    // own thread-affine scope lifecycle (built off the clock,
                    // destroyed on this thread after the verify gate).
                    const llaminar::v2::OpenMPTeamForkCell *worker_stack_at_entry =
                        llaminar::v2::openMPTeamForkOwnerStack;
                    std::optional<llaminar::v2::OpenMPTeamForkScope> scope;
                    scope.emplace();
                    worker_ready.store(true, std::memory_order_release);
                    while (!start_gate.load(std::memory_order_acquire))
                        std::this_thread::yield();
                    for (std::uint64_t note = 0; note < kMeasuredNotes; ++note)
                        llaminar::v2::noteOpenMPTeamFork();
                    worker_notes_done.store(true, std::memory_order_release);
                    while (!verify_gate.load(std::memory_order_acquire))
                        std::this_thread::yield();
                    EXPECT_EQ(scope->forks(), kMeasuredNotes);
                    scope.reset();
                    EXPECT_EQ(llaminar::v2::openMPTeamForkOwnerStack,
                              worker_stack_at_entry)
                        << "the worker's owner stack must be restored exactly";
                });
                while (!worker_ready.load(std::memory_order_acquire))
                    std::this_thread::yield();
            }
            else
            {
                if (owners >= 1)
                    outer.emplace();
                if (owners >= 2)
                    inner.emplace();
            }
            // The timed interval is the AMORTIZED CONCURRENT WALL TIME
            // INCLUDING RENDEZVOUS: it opens with the start-gate publication
            // and closes after the competing owner's completion is observed.
            // Readiness gates permit concurrency; they do not prove scheduler
            // overlap, so the interval is honestly labeled as wall time
            // including the rendezvous, never as a literal notes-only span.
            // Verification, join and teardown happen after the clock stops.
            const auto cost_start = std::chrono::steady_clock::now();
            if (concurrent)
                start_gate.store(true, std::memory_order_release);
            for (std::uint64_t note = 0; note < kMeasuredNotes; ++note)
                llaminar::v2::noteOpenMPTeamFork();
            if (concurrent)
                while (!worker_notes_done.load(std::memory_order_acquire))
                    std::this_thread::yield();
            const auto cost_stop = std::chrono::steady_clock::now();
            if (concurrent)
                verify_gate.store(true, std::memory_order_release);
            if (concurrent)
            {
                EXPECT_EQ(main_owner->forks(), kMeasuredNotes);
                main_owner.reset(); // Destroyed on the thread that owns it.
                // EXACT TLS STACK IDENTITY after the main owner's teardown.
                EXPECT_EQ(llaminar::v2::openMPTeamForkOwnerStack, stack_at_entry)
                    << "the original owner stack must be restored exactly";
                worker.join(); // Verified and destroyed on its own thread.
            }
            else
            {
                EXPECT_EQ(outer ? outer->forks() : kMeasuredNotes, kMeasuredNotes)
                    << "owners " << owners;
                if (inner)
                    EXPECT_EQ(inner->forks(), kMeasuredNotes) << "owners " << owners;
                // The total is checked BEFORE the teardown probe so the
                // probe's own fork stays out of the measured figure.
                EXPECT_EQ(llaminar::v2::openMPTeamForkSnapshot() - before,
                          kMeasuredNotes)
                    << "nested owners " << owners;
                if (outer)
                {
                    // LIFO: the inner dies first (present only for
                    // owners>=2), then the probe proves stack identity by
                    // attribution, then the outer dies.
                    inner.reset();
                    {
                        llaminar::v2::OpenMPTeamForkScope restored;
                        auto local_idle = [] {};
                        OMP_WORKSHARE_REGION(local_idle);
                        EXPECT_EQ(restored.forks(), 1u);
                        // STACK IDENTITY, not just a self-count: the probe
                        // fork must attribute to BOTH the fresh scope and
                        // the surviving outer. A lost or wrong previous link
                        // would leave the outer at kMeasuredNotes and fail
                        // here.
                        EXPECT_TRUE(outer.has_value());
                        EXPECT_EQ(outer->forks(), kMeasuredNotes + 1)
                            << "the probe must walk restored AND the surviving outer";
                    }
                    outer.reset(); // Only after the inner is gone and verified.
                }
                else
                {
                    // owners==0: nothing was alive; GENUINE no-owner - the
                    // restoration probe never dereferences a disengaged
                    // optional (the UB this branch used to carry).
                    inner.reset();
                }
                // EXACT TLS STACK IDENTITY after the teardown: the owner
                // stack pointer must be the ORIGINAL saved pointer, byte for
                // byte - a stale or extra link would fail here. The global
                // total is NOT the witness: it advances once per fork
                // whatever the stack holds.
                EXPECT_EQ(llaminar::v2::openMPTeamForkOwnerStack, stack_at_entry)
                    << "the original owner stack must be restored exactly";
            }
            const auto cost_ns = static_cast<double>(
                std::chrono::duration_cast<std::chrono::nanoseconds>(cost_stop - cost_start)
                    .count());
            // The restoration probe counts ONLY when its guard actually
            // executes it (non-concurrent with an engaged outer); measure(0,
            // false) skips the probe, so its total is exactly the notes.
            const std::uint64_t expected_total =
                kMeasuredNotes * (concurrent ? 2u : 1u) +
                (concurrent ? 0u : (outer ? 1u : 0u));
            EXPECT_EQ(llaminar::v2::openMPTeamForkSnapshot() - before, expected_total);
            return cost_ns / static_cast<double>(kMeasuredNotes);
        };
        const std::pair<const char *, double> variants[] = {
            {"no_owner", measure(0, false)},
            {"one_owner", measure(1, false)},
            {"two_nested_owners", measure(2, false)},
            {"concurrent_two_threads", measure(1, true)}};
        for (const auto &[label, ns_per_note] : variants)
        {
            PerfStatsCollector::addCounter(
                "omp", std::string("openmp_team_fork_note_self_cost_") + label,
                ns_per_note, "evidence", "host",
                {{"basis", std::to_string(kMeasuredNotes) +
                               "_notes_per_measuring_thread_per_owner_latency_"
                               "amortized_concurrent_wall_time_including_rendezvous_"
                               "scope_config_built_off_clock"}});
            std::cout << "[evidence] openmp_team_fork_note_self_cost_" << label << "="
                      << ns_per_note << " ns per owner note, amortized concurrent wall time "
                                        "including rendezvous over " << kMeasuredNotes
                      << " notes per measuring thread (per-owner latency with the production "
                         "owner walk; readiness permits concurrency and does not prove "
                         "scheduler overlap; earlier no-walk figures do not transfer)" << std::endl;
            EXPECT_LT(ns_per_note, 100.0) << label;
        }
    }
    // The evidence configuration has ended: the guards restored the caller's
    // environment values, and this reload runs AFTER their destruction, so
    // the authorities describe exactly the restored environment and no export
    // selection outlives the test.
    mutableDebugEnv().reload();
    PerfStatsCollector::reloadConfigurationFromEnvironment();
    PerfStatsCollector::reset();
}

/** Credentials and rank-local hardware policy must not become cluster-wide settings. */
TEST(Test__MPIBootstrap, DiagnosticPropagationExcludesSecretsAndDeviceLocalPolicy)
{
    for (const char *name : {"LLAMINAR_AZURE_AUTH_TOKEN", "AZURE_CLIENT_SECRET",
                            "CUDA_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES",
                            "OMP_NUM_THREADS", "LLAMINAR_MOE_GPU_CACHE_EXPERTS_PER_LAYER"})
    {
        ScopedLaunchEnvironment environment(name, "not-a-portable-diagnostic");
        llaminar2::MPILaunchConfig config;
        config.hostfile = "/cluster/hosts";
        const auto command = buildCommand(config, {});
        EXPECT_FALSE(contains(command, name));
        EXPECT_FALSE(contains(command, "not-a-portable-diagnostic"));
    }
}

/** Explicit intent may never carry the unresolved NUMA sentinel into launch. */
TEST(Test__MPIBootstrap, ExplicitCpuPlacementRejectsUnknownNode)
{
    llaminar2::OrchestrationConfig config;
    config.device_for_this_rank = llaminar2::GlobalDeviceAddress::cpu();
    config.device_for_this_rank_numa_explicit = true;
    EXPECT_THROW(
        llaminar2::MPIBootstrapPhase::resolveInferenceNUMANodes(
            config, llaminar2::DeviceManager::instance(), {}),
        std::invalid_argument);
}

TEST(Test__MPIBootstrap, SocketMappedCpuTPReservesFullSocketProcessingElements)
{
    llaminar2::CPUTopology topology;
    topology.num_sockets = 2;
    topology.physical_cores = 56;
    topology.logical_cores = 112;
    topology.cores_per_socket = 28;
    topology.threads_per_core = 2;
    topology.numa_nodes = 2;
    topology.hyperthreading = true;

    llaminar2::MPILaunchConfig config;
    config.num_procs = 2;
    config.bind_to_socket = true;
    config.map_by_socket = true;
    config.omp_threads_per_rank = 28;

    const auto cmd = buildCommand(config, topology);

    EXPECT_TRUE(contains(cmd, "--bind-to"));
    EXPECT_TRUE(contains(cmd, "core"))
        << "OpenMPI requires bind-to core when mapping PE=N cores/rank";
    EXPECT_TRUE(contains(cmd, "--map-by"));
    EXPECT_TRUE(contains(cmd, "socket:PE=28"))
        << "plain socket mapping binds only one physical core plus its SMT sibling "
           "on the observed dual-socket OpenMPI setup";
}

TEST(Test__MPIBootstrap, SingleRankAllCoreLaunchUsesSlotProcessingElements)
{
    llaminar2::CPUTopology topology;
    topology.num_sockets = 2;
    topology.physical_cores = 56;
    topology.logical_cores = 112;
    topology.cores_per_socket = 28;
    topology.threads_per_core = 2;
    topology.numa_nodes = 2;
    topology.hyperthreading = true;

    llaminar2::MPILaunchConfig config;
    config.num_procs = 1;
    config.bind_to_socket = true;
    config.map_by_socket = true;
    config.omp_threads_per_rank = 56;

    const auto cmd = buildCommand(config, topology);

    EXPECT_TRUE(contains(cmd, "core"));
    EXPECT_TRUE(contains(cmd, "slot:PE=56"))
        << "a single all-core CPU rank cannot fit inside one socket mapping";
}

/**
 * @brief CPU overlay selectors contribute every requested NUMA node pre-MPI.
 *
 * This is the focused regression for a heterogeneous overlay whose four ROCm
 * participants were local to NUMA 1 while `cpu:0,cpu:1` was accidentally
 * treated as unresolved CPU ordinals.  Bootstrap then pinned both ranks to
 * NUMA 1 and made certified first-touch allocation on NUMA 0 impossible.
 */
TEST(Test__MPIBootstrap, OverlayCpuShortSelectorsDriveBootstrapNumaSet)
{
    llaminar2::OrchestrationConfig config =
        llaminar2::OrchestrationConfig::defaults();
    config.domain_definitions.push_back(
        llaminar2::DomainDefinition::parse(
            "cpu_tier=cpu:0,cpu:1;scope=auto;backend=upi"));

    llaminar2::CPUTopology topology;
    topology.numa_nodes = 2;

    const auto nodes =
        llaminar2::MPIBootstrapPhase::resolveInferenceNUMANodes(
            config,
            llaminar2::DeviceManager::instance(),
            topology);

    EXPECT_EQ(nodes, (std::set<int>{0, 1}));
}

/** @test Opaque wrappers retain their complete original parser boundary after MPI re-exec. */
TEST(Test__MPIBootstrap, ObserverOpaqueArgvRetainsOriginalSeparatorAndLaunchFlags)
{
    using namespace llaminar2;
    std::vector<std::string> original = {"observer", "--workload", "/fixture/work load.json",
        "--output", "/fixture/output", "--", "--model", "/synthetic/main.gguf",
        "--mpi-procs", "1", "--device", "localhost:0:cpu:0", "--mpi-procs=1", "--mpi-dry-run"};
    std::vector<char *> argv;
    for (auto &value : original) argv.push_back(value.data());
    MPILaunchConfig launch;
    launch.num_procs = 1;
    launch.cpu_set = "0-3"; // Authored launch metadata; no actual affinity/discovery.
    launch.bind_to_socket = launch.map_by_socket = false;
    const auto command = MPIBootstrap::buildMPIRunCommand(static_cast<int>(argv.size()), argv.data(), launch, {},
        MPIBootstrapArgumentPolicy::PreserveOriginalArguments);
    const auto binary = std::find(command.begin(), command.end(), "observer");
    ASSERT_NE(binary, command.end());
    EXPECT_EQ((std::vector<std::string>(binary, command.end())), original);
    const auto rank_flag = std::find(command.begin(), binary, "-np");
    ASSERT_NE(rank_flag, binary);
    ASSERT_NE(rank_flag + 1, binary);
    EXPECT_EQ(*(rank_flag + 1), "1");
    const auto cpus = std::find(command.begin(), binary, "--cpu-set");
    ASSERT_NE(cpus, binary);
    ASSERT_NE(cpus + 1, binary);
    EXPECT_EQ(*(cpus + 1), "0-3");
}

/** @test Ordinary callers retain configuration while consuming only launch actions. */
TEST(Test__MPIBootstrap, ObserverArgumentExtensionPreservesLegacyDefaultFiltering)
{
    using namespace llaminar2;
    std::vector<std::string> original = {"observer", "--workload", "work.json", "--output", "out", "--",
        "--mpi-procs", "1", "--mpi-procs=1", "--mpi-bootstrap", "--mpi-dry-run", "--device", "cpu:0"};
    std::vector<char *> argv;
    for (auto &value : original) argv.push_back(value.data());
    const auto unchanged = MPIBootstrap::buildMPIRunCommand(static_cast<int>(argv.size()), argv.data(), {}, {});
    EXPECT_EQ(unchanged, MPIBootstrap::buildMPIRunCommand(static_cast<int>(argv.size()), argv.data(), {}, {},
        MPIBootstrapArgumentPolicy::FilterLaunchArguments));
    const auto binary = std::find(unchanged.begin(), unchanged.end(), "observer");
    ASSERT_NE(binary, unchanged.end());
    EXPECT_EQ((std::vector<std::string>(binary, unchanged.end())),
        (std::vector<std::string>{"observer", "--workload", "work.json", "--output", "out", "--",
            "--mpi-procs", "1", "--mpi-procs=1", "--device", "cpu:0"}));
    EXPECT_THROW((void)MPIBootstrap::buildMPIRunCommand(static_cast<int>(argv.size()), argv.data(), {}, {},
        static_cast<MPIBootstrapArgumentPolicy>(99)), std::invalid_argument);
}

/** @test The actual public parser sees the same explicit CPU config after retained re-exec. */
TEST(Test__MPIBootstrap, ObserverCPUConfigAndCanonicalIntentSurviveRetainedArgv)
{
    using namespace llaminar2;
    std::vector<std::string> engine = {"observer", "--model", "/synthetic/nonexistent-qfn.gguf",
        "--backend", "host", "--device", "localhost:0:cpu:0", "--mpi-procs", "1", "--no-mtp"};
    std::vector<char *> engine_argv;
    for (auto &value : engine) engine_argv.push_back(value.data());
    const auto original = OrchestrationConfigParser{}.parseArgs(static_cast<int>(engine_argv.size()), engine_argv.data());
    EXPECT_EQ(original.n_threads, -1);
    EXPECT_EQ(original.mpi_procs, 1);
    ASSERT_TRUE(original.device_for_this_rank);
    EXPECT_TRUE(original.device_for_this_rank_numa_explicit);
    EXPECT_EQ(original.device_for_this_rank->numa_node, 0);
    EXPECT_EQ(MPIBootstrapPhase::classifyDeviceIntent(original), BootstrapDeviceIntent::CpuOnly);
    std::vector<std::string> wrapper = {"observer", "--workload", "fixture.json", "--output", "out", "--"};
    wrapper.insert(wrapper.end(), engine.begin() + 1, engine.end());
    std::vector<char *> wrapper_argv;
    for (auto &value : wrapper) wrapper_argv.push_back(value.data());
    MPILaunchConfig launch;
    launch.num_procs = 1;
    const auto command = MPIBootstrap::buildMPIRunCommand(static_cast<int>(wrapper_argv.size()), wrapper_argv.data(), launch, {},
        MPIBootstrapArgumentPolicy::PreserveOriginalArguments);
    const auto binary = std::find(command.begin(), command.end(), "observer");
    ASSERT_NE(binary, command.end());
    const auto separator = std::find(binary, command.end(), "--");
    ASSERT_NE(separator, command.end());
    std::vector<std::string> restored{*binary};
    restored.insert(restored.end(), separator + 1, command.end());
    std::vector<char *> restored_argv;
    for (auto &value : restored) restored_argv.push_back(value.data());
    const auto actual = OrchestrationConfigParser{}.parseArgs(static_cast<int>(restored_argv.size()), restored_argv.data());
    EXPECT_EQ(serializeOrchestrationConfig(actual), serializeOrchestrationConfig(original));
    EXPECT_EQ(MPIBootstrapPhase::classifyDeviceIntent(actual), BootstrapDeviceIntent::CpuOnly);
}

/** @test A CPU continuation with a GPU domain cannot enter the CPU-only tool bootstrap. */
TEST(Test__MPIBootstrap, ObserverCanonicalIntentIncludesNestedAcceleratorDeclarations)
{
    using namespace llaminar2;
    auto config = OrchestrationConfig::defaults();
    config.device_for_this_rank = GlobalDeviceAddress::cpu(0);
    config.device_for_this_rank_numa_explicit = true;
    EXPECT_EQ(MPIBootstrapPhase::classifyDeviceIntent(config), BootstrapDeviceIntent::CpuOnly);
    config.domain_definitions.push_back(DomainDefinition::parse("compute=cuda:0;scope=rank_local;backend=nccl"));
    EXPECT_EQ(MPIBootstrapPhase::classifyDeviceIntent(config), BootstrapDeviceIntent::Accelerator);
    config.domain_definitions.clear();
    config.domain_definitions.push_back(DomainDefinition::parse("compute=rocm:0;scope=rank_local;backend=rccl"));
    EXPECT_EQ(MPIBootstrapPhase::classifyDeviceIntent(config), BootstrapDeviceIntent::Accelerator);
}

/** @brief Native MPI options end before any opaque application argument is inspected. */
TEST(Test__MPIBootstrap, NativeOptionBoundaryPreservesOpaqueRuntimeNamedApplicationArguments)
{
    using namespace llaminar2;
    char binary[]="frontend", option[]="--tmpdir", value[]="application-owned-value";
    char *argv[]={binary,option,value};
    const auto command=MPIBootstrap::buildMPIRunCommand(3,argv,{},{});
    const auto application=std::find(command.begin(),command.end(),"frontend");
    ASSERT_NE(application,command.end());
    ASSERT_NE(application,command.begin());
    EXPECT_EQ(*(application-1),"--");
    EXPECT_EQ((std::vector<std::string>(application,command.end())),
              (std::vector<std::string>{binary,option,value}));
}

/** @test Ordinary child parsing preserves explicit rank intent and unspecified auto. */
TEST(Test__MPIBootstrap, ApplicationRankConfigurationSurvivesFilteredReexec)
{
    using namespace llaminar2;
    for (const int count : {0, 1, 4})
    {
        for (const bool equals : {false, true})
        {
            SCOPED_TRACE(::testing::Message() << "count=" << count << " equals=" << equals);
            std::vector<std::string> original = {"llaminar2", "--model", "/synthetic/main.gguf",
                "--backend", "host", "--no-mtp"};
            if (count != 0)
            {
                original.push_back(equals ? "--mpi-procs=" + std::to_string(count) : "--mpi-procs");
                if (!equals) original.push_back(std::to_string(count));
            }
            const auto parse = [](std::vector<std::string> arguments)
            {
                std::vector<char *> argv;
                for (auto &argument : arguments) argv.push_back(argument.data());
                return OrchestrationConfigParser{}.parseArgs(static_cast<int>(argv.size()), argv.data());
            };
            const auto before = parse(original);
            auto launched = original;
            launched.insert(launched.end(), {"--mpi-bootstrap", "--mpi-dry-run"});
            std::vector<char *> argv;
            for (auto &argument : launched) argv.push_back(argument.data());
            MPILaunchConfig launch;
            launch.num_procs = count == 0 ? 2 : count;
            const auto command = MPIBootstrap::buildMPIRunCommand(
                static_cast<int>(argv.size()), argv.data(), launch, {});
            const auto child = std::find(command.begin(), command.end(), "llaminar2");
            ASSERT_NE(child, command.end());
            const std::vector<std::string> restored(child, command.end());
            EXPECT_EQ(restored, original);
            const auto after = parse(restored);
            EXPECT_EQ(after.mpi_procs, count);
            EXPECT_EQ(serializeOrchestrationConfig(after), serializeOrchestrationConfig(before));
            const auto ranks = std::find(command.begin(), child, "-np");
            ASSERT_NE(ranks, child);
            ASSERT_NE(ranks + 1, child);
            EXPECT_EQ(*(ranks + 1), std::to_string(launch.num_procs));
        }
    }
}

/** @test A retained explicit rank count does not launch recursively inside MPI. */
TEST(Test__MPIBootstrap, ExistingMPIKeepsExplicitRankConfigurationWithoutRelaunch)
{
    using namespace llaminar2;
    /** @brief Restore the original typed environment snapshot after injected MPI metadata. */
    struct RestoreBootstrapSnapshot
    {
        MPIBootstrapEnvConfig original = debugEnv().mpi_bootstrap;
        ~RestoreBootstrapSnapshot() { mutableDebugEnv().mpi_bootstrap = original; }
    } restore_snapshot;
    ScopedLaunchEnvironment rank("OMPI_COMM_WORLD_RANK", "1");
    ScopedLaunchEnvironment size("OMPI_COMM_WORLD_SIZE", "4");
    mutableDebugEnv().mpi_bootstrap.reload();
    ASSERT_TRUE(MPIBootstrap::detectMPIEnvironment().is_mpi_process);
    auto config = OrchestrationConfig::defaults();
    config.mpi_procs = 4;
    config.device_for_this_rank = GlobalDeviceAddress::cpu(0);
    char binary[] = "llaminar2";
    char *argv[] = {binary};
    const auto result = MPIBootstrapPhase{}.execute(config, 1, argv);
    EXPECT_EQ(result.action, BootstrapResult::Action::CONTINUE);
    EXPECT_EQ(config.mpi_procs, 4);
}

/** @test Invalid native place publications cannot reach exec or worker admission. */
TEST(Test__MPIBootstrap, NativeCorePlaceReceiptRejectsMalformedAndIncompatiblePublication)
{
    using namespace llaminar2;
    static_assert(!std::is_default_constructible_v<NativeOpenMPCorePlaces>);
    static_assert(!std::is_copy_constructible_v<NativeOpenMPCorePlaces>);
    char pattern[]="/tmp/llaminar-native-receipt-XXXXXX";
    const char *root=::mkdtemp(pattern);
    ASSERT_NE(root,nullptr);
    struct Cleanup {std::filesystem::path path;~Cleanup(){std::filesystem::remove_all(path);}} cleanup{root};
    const auto process=std::to_string(::getpid());
    const std::vector<std::vector<int>> canonical{{0},{1}};
    for(const auto &receipt:std::vector<std::string>{"", "v2;1;{0}", "v1;;{0}",
            "v1;bad;{0}", "v1;-1;{0}", "v1;"+process+";", "v1;"+process+";{-1}"})
    {
        SCOPED_TRACE(receipt);
        EXPECT_THROW(NativeCPUPlacement::validateCapacityReceipt(receipt,canonical,::getpid(),root),std::runtime_error);
    }
    EXPECT_NO_THROW(NativeCPUPlacement::validateCapacityReceipt(
        "v1;"+process+";{0},{1}",canonical,::getpid(),root));
    // Matching native/receipt bytes are insufficient when their capacity order is wrong.
    EXPECT_THROW(NativeCPUPlacement::validateCapacityReceipt(
        "v1;"+process+";{1},{0}",{{1},{0}},::getpid(),root),std::runtime_error);
}

/** @test Replay recorded GB10 kernel observations through the production place resolver. */
TEST(Test__MPIBootstrap, RecordedGB10NativePlaceOrderReplay)
{
    using namespace llaminar2;
    char pattern[]="/tmp/llaminar-native-place-replay-XXXXXX";
    const char *root=::mkdtemp(pattern);
    ASSERT_NE(root,nullptr);
    struct Cleanup {std::filesystem::path path;~Cleanup(){std::filesystem::remove_all(path);}} cleanup{root};
    ASSERT_EQ(test::kRecordedCorePlacePolicyVersion,1);
    std::vector<int> allowed;
    std::vector<std::vector<int>> expected;
    for(const auto &cpu:test::kRecordedGB10CoreCapacities)
    {
        const auto directory=std::filesystem::path(root)/("cpu"+std::to_string(cpu.cpu));
        std::filesystem::create_directories(directory/"topology");
        {std::ofstream output(directory/"cpu_capacity");output<<cpu.capacity<<'\n';ASSERT_TRUE(output.good());}
        {std::ofstream output(directory/"topology/thread_siblings_list");output<<cpu.physical_sibling<<'\n';ASSERT_TRUE(output.good());}
        allowed.push_back(cpu.cpu);
    }
    for(const int cpu:test::kRecordedGB10CorePlaceOrder)expected.push_back({cpu});
    for(int repeat=0;repeat<2;++repeat)
        EXPECT_EQ(NativeCPUPlacement::capacityOrderedCorePlaces(allowed,root),expected);
    const auto process=std::to_string(::getpid());
    const std::string encoding="{19},{15},{16},{17},{18},{5},{6},{7},{8},{9},{10},{11},{12},{13},{14},{0},{1},{2},{3},{4}";
    EXPECT_NO_THROW(NativeCPUPlacement::validateCapacityReceipt("v1;"+process+';'+encoding,expected,::getpid(),root));
    std::vector<std::vector<int>> stale;
    std::string stale_encoding;
    for(const int cpu:allowed)
    {
        stale.push_back({cpu});
        if(!stale_encoding.empty())stale_encoding+=',';
        stale_encoding+='{'+std::to_string(cpu)+'}';
    }
    EXPECT_THROW(NativeCPUPlacement::validateCapacityReceipt("v1;"+process+';'+stale_encoding,stale,::getpid(),root),std::runtime_error);
}

/** @test Late and foreign callers cannot reopen original entry or publish another exec. */
TEST(Test__MPIBootstrap, NativeCorePlaceEntryRejectsForeignAndLateCallers)
{
    using namespace llaminar2;
    ASSERT_TRUE(MPISessionOwner::sharedDirectory().has_value());
    const auto environment=[] {
        std::vector<std::optional<std::string>> values;
        for(const char *name:{"OMP_PLACES","OMP_PROC_BIND","LLAMINAR_NATIVE_CORE_PLACES"})
        {
            const char *value=std::getenv(name);
            values.push_back(value ? std::optional<std::string>(value) : std::nullopt);
        }
        return values;
    };
    const auto before=environment();
    char program[]="late-native-entry";
    char *argv[]={program};
    const auto late=MPISessionOwner::enterProcess(1,argv);
    ASSERT_TRUE(late.has_value());
    EXPECT_EQ(*late,125);
    std::optional<int> foreign;
    {
        std::jthread worker([&]{foreign=MPISessionOwner::enterProcess(1,argv);});
    }
    ASSERT_TRUE(foreign.has_value());
    EXPECT_EQ(*foreign,125);
    EXPECT_EQ(environment(),before);
}

/** @test Core places precede exec on every launch path; explicit placement remains authoritative. */
TEST(Test__MPIBootstrap, PortableOpenMPPlacementPreservesRankLocalWorkerGeometry)
{
    using namespace llaminar2;
    ScopedLaunchEnvironment places("OMP_PLACES", nullptr);
    ScopedLaunchEnvironment binding("OMP_PROC_BIND", nullptr);
    ScopedLaunchEnvironment threads("OMP_NUM_THREADS", "7");
    for (const char *hostfile : {"", "/cluster/hosts"})
    for (const int ranks : {1, 2, 4, 8})
    for (const char *selected_places : {"cores", "threads", "sockets", "{2},{4}"})
    for (const char *selected_binding : {"close", "spread", "false"})
    {
        MPILaunchConfig launch;
        EXPECT_EQ(launch.omp_places, "cores");
        launch.hostfile = hostfile;
        launch.num_procs = ranks;
        launch.omp_places = selected_places;
        launch.omp_proc_bind = selected_binding;
        MPIBootstrap::configureOpenMPPlacement(launch);
        EXPECT_STREQ(std::getenv("OMP_PLACES"), selected_places);
        EXPECT_STREQ(std::getenv("OMP_PROC_BIND"), selected_binding);
        EXPECT_STREQ(std::getenv("OMP_NUM_THREADS"), "7");
        const auto command = buildCommand(launch, {});
        for (const char *name : {"OMP_PLACES", "OMP_PROC_BIND"})
        {
            const auto at = std::find(command.begin(), command.end(), name);
            ASSERT_NE(at, command.end());
            ASSERT_NE(at, command.begin());
            EXPECT_EQ(*(at - 1), "-x");
            EXPECT_EQ(std::count(command.begin(), command.end(), name), 1);
        }
        EXPECT_FALSE(contains(command, "OMP_NUM_THREADS"));
        EXPECT_TRUE(launch.bind_to_socket);
        EXPECT_TRUE(launch.map_by_socket);
    }
    MPILaunchConfig invalid;
    invalid.omp_places.clear();
    EXPECT_THROW(MPIBootstrap::configureOpenMPPlacement(invalid), std::invalid_argument);
}

/** @test Typed original observations preserve sparse nodes and actual logical indices at the MPI boundary. */
TEST(Test__MPIBootstrap, ObservedNUMAMappingPreservesSparseMembershipAndOriginalArguments)
{
    using namespace llaminar2;
    for (const auto &observed : {std::vector<MPICorePlacement>{{9,4,{101,501}}},
        std::vector<MPICorePlacement>{{4,7,{103,503}},{42,11,{301,701}}},
        std::vector<MPICorePlacement>{{3,2,{107}},{11,8,{129,131}},{70,19,{901,903,905}}}})
    {
        std::vector<int> selected;
        for (const auto &core : observed) selected.push_back(core.os_numa_node);
        MPILaunchConfig launch;
        launch.numa_mapping = MPINUMAMapping::fromObservedPlacement(observed, selected);
        launch.num_procs = static_cast<int>(selected.size());
        launch.bind_to_socket = launch.map_by_socket = false;
        std::vector<std::string> arguments{"original-native-owner", "--gtest_filter=Exact/OriginalBody"};
        std::vector<char *> argv;for (auto &argument : arguments) argv.push_back(argument.data());
        const auto command = MPIBootstrap::buildMPIRunCommand(argv.size(), argv.data(), launch, {},
            MPIBootstrapArgumentPolicy::PreserveOriginalArguments, "/fixture/original-observed-numa.rankfile");
        const auto value = [&](const std::string &option) -> std::string {
            const auto found = std::find(command.begin(), command.end(), option);
            EXPECT_NE(found, command.end());
            return found == command.end() || std::next(found) == command.end() ? "" : *std::next(found);
        };
        EXPECT_EQ(value("-np"), std::to_string(selected.size()));
        EXPECT_EQ(value("--rankfile"), "/fixture/original-observed-numa.rankfile");
        EXPECT_EQ(std::count(command.begin(), command.end(), "--rankfile"), 1);
        EXPECT_FALSE(contains(command, "--map-by"));
        EXPECT_FALSE(contains(command, "--cpu-set"));
        EXPECT_FALSE(contains(command, "--bind-to"));
        EXPECT_EQ(value("rmaps_rank_file_physical"), "0");
        EXPECT_EQ(value("hwloc_base_use_hwthreads_as_cpus"), "0");
        EXPECT_EQ(std::count(launch.numa_mapping->rankFileContents().begin(),
            launch.numa_mapping->rankFileContents().end(), '\n'), selected.size());
        EXPECT_THROW((void)MPIBootstrap::buildMPIRunCommand(argv.size(),argv.data(),launch,{}),std::invalid_argument);
        EXPECT_EQ(command.back(), arguments.back());
        auto wrong = launch; wrong.num_procs = launch.num_procs + 1;
        EXPECT_THROW((void)MPIBootstrap::buildMPIRunCommand(argv.size(),argv.data(),wrong,{}),std::invalid_argument);
        wrong=launch;wrong.hostfile="original-hostfile";
        EXPECT_THROW((void)MPIBootstrap::buildMPIRunCommand(argv.size(),argv.data(),wrong,{}),std::invalid_argument);
        wrong=launch;wrong.map_by_socket=true;
        EXPECT_THROW((void)MPIBootstrap::buildMPIRunCommand(argv.size(),argv.data(),wrong,{}),std::invalid_argument);
    }
}

/** @test OS NUMA/PU labels cannot masquerade as logical cores or erase partial/contradictory observations. */
TEST(Test__MPIBootstrap, ObservedNUMAMappingConvertsNonmatchingOSAndLogicalIndices)
{
    using namespace llaminar2;
    const std::vector<MPICorePlacement> actual{
        {42,7,{301,701}}, {4,11,{103,503}}, {42,2,{305,705}}, {99,19,{901,1001}}};
    const auto logical=NUMATopology::logicalMPICoresForNUMANodes(actual,std::array{42,4});
    EXPECT_EQ(logical.at(42),(std::vector<int>{2,7})); EXPECT_EQ(logical.at(4),(std::vector<int>{11}));
    MPILaunchConfig launch;
    launch.numa_mapping=MPINUMAMapping::fromObservedPlacement(actual,std::array{42,4});
    launch.num_procs=2;launch.bind_to_socket=launch.map_by_socket=false;
    const auto command=buildCommand(launch,{});
    const auto where=std::find(command.begin(),command.end(),"--rankfile");
    ASSERT_NE(where,command.end());ASSERT_NE(std::next(where),command.end());
    EXPECT_EQ(*std::next(where),"/fixture/original-observed-numa.rankfile");
    EXPECT_EQ(launch.numa_mapping->rankFileContents(),"rank 0=localhost slot=11\nrank 1=localhost slot=2,7\n");
    EXPECT_EQ(launch.numa_mapping->rankFileContents().find("301"),std::string::npos);
    EXPECT_EQ(launch.numa_mapping->rankFileContents().find("701"),std::string::npos);
    EXPECT_EQ(std::find(command.begin(),command.end(),"301"),command.end());
    EXPECT_THROW((void)MPINUMAMapping::fromObservedPlacement({},std::array{42}),std::invalid_argument);
    EXPECT_THROW((void)MPINUMAMapping::fromObservedPlacement(actual,std::array{8}),std::invalid_argument);
    EXPECT_THROW((void)MPINUMAMapping::fromObservedPlacement(actual,std::array{42,42}),std::invalid_argument);
    auto invalid=actual;invalid.back().logical_core_index=7;
    EXPECT_THROW((void)MPINUMAMapping::fromObservedPlacement(invalid,std::array{42}),std::invalid_argument);
    invalid=actual;invalid.back().os_cpu_ids={301};
    EXPECT_THROW((void)MPINUMAMapping::fromObservedPlacement(invalid,std::array{42}),std::invalid_argument);
    invalid=actual;invalid.back().os_numa_node=-1;
    EXPECT_THROW((void)MPINUMAMapping::fromObservedPlacement(invalid,std::array{42}),std::invalid_argument);
    invalid=actual;invalid.back().os_cpu_ids.clear();
    EXPECT_THROW((void)MPINUMAMapping::fromObservedPlacement(invalid,std::array{42}),std::invalid_argument);
}

/** @test Native parser truncation is rejected before an incomplete physical binding can launch. */
TEST(Test__MPIBootstrap, ObservedNUMAMappingRejectsNativeRankfileTruncation)
{
    using namespace llaminar2;
    std::vector<MPICorePlacement> sparse;
    for (int core = 0; core < 40; ++core)
        sparse.push_back({42, core * 2, {1000 + core}});
    EXPECT_THROW((void)MPINUMAMapping::fromObservedPlacement(sparse, std::array{42}), std::invalid_argument);
    MPILaunchConfig regular;
    char program[] = "original-owner"; char *argv[]{program};
    EXPECT_THROW((void)MPIBootstrap::buildMPIRunCommand(1, argv, regular, {},
        MPIBootstrapArgumentPolicy::PreserveOriginalArguments, "/unowned/undeclared.rankfile"), std::invalid_argument);
}

/** @test One package may expose multiple exact nodes; missing facts never become guessed ownership. */
TEST(Test__MPIBootstrap, ObservedNUMAMappingRequiresExactCPUResourceFacts)
{
    using namespace llaminar2;
    HardwareInventory hardware;
    hardware.cpu_sockets={{.socket_id=0,.numa_node=4,.physical_cores={103,301},
        .memory_bytes=100,.available_memory_bytes=60}};
    hardware.cpu_last_level_cache_bytes={{0,32u<<20}};
    EXPECT_THROW((void)hardware.cpuDevice(4),std::invalid_argument);
    EXPECT_THROW((void)hardware.cpuDevice(),std::invalid_argument);
    hardware.cpu_numa_devices={
        {4,{.type=ComputeBackendType::CPU,.numa_node=4,.total_memory_bytes=100,.free_memory_bytes=60,
            .compute_units=2,.last_level_cache_bytes=32u<<20}},
        {42,{.type=ComputeBackendType::CPU,.numa_node=42,.total_memory_bytes=200,.free_memory_bytes=170,
            .compute_units=2,.last_level_cache_bytes=32u<<20}}};
    EXPECT_EQ(hardware.cpuDevice(42).total_memory_bytes,200u);
    EXPECT_EQ(hardware.cpuDevice(42).numa_node,42);
    EXPECT_EQ(hardware.cpuDevice().total_memory_bytes,300u);
    EXPECT_EQ(hardware.cpuDevice().compute_units,4);
    EXPECT_EQ(hardware.cpuDevice().last_level_cache_bytes,32u<<20);
    EXPECT_THROW((void)hardware.cpuDevice(17),std::invalid_argument);
    hardware.cpu_numa_devices.at(42).numa_node=4;
    EXPECT_THROW((void)hardware.cpuDevice(4),std::invalid_argument);
    hardware.cpu_numa_devices.at(42).numa_node=42;
    hardware.cpu_numa_devices.at(42).free_memory_bytes=201;
    EXPECT_THROW((void)hardware.cpuDevice(42),std::invalid_argument);
}

/**
 * @test Local rankfiles name their allocation without a relative-host dependency.
 * Repeated one/two/three-rank construction retains exact observed core lists;
 * neither ordinary local nor hostfile launch inherits this local-only option.
 */
TEST(Test__MPIBootstrap, ObservedNUMAMappingNamesLocalHostInRankfile)
{
    using namespace llaminar2;
    const std::vector<MPICorePlacement> observed{{4,0,{100}},{42,2,{200}},{70,4,{300}}};
    const std::array all_nodes{4,42,70};
    for (const size_t count : {2u,1u,3u,2u})
    {
        MPILaunchConfig launch;
        launch.numa_mapping = MPINUMAMapping::fromObservedPlacement(observed,
            std::span<const int>(all_nodes.data(),count));
        launch.num_procs = static_cast<int>(count);
        launch.bind_to_socket = launch.map_by_socket = false;
        const auto original_rankfile = launch.numa_mapping->rankFileContents();
        const auto command = buildCommand(launch,{});
        const auto rankfile = std::find(command.begin(),command.end(),"--rankfile");
        ASSERT_NE(rankfile,command.end());
        EXPECT_FALSE(contains(command,"--host"));
        EXPECT_EQ(std::count(command.begin(),command.end(),"--rankfile"),1);
        const auto ranks = std::find(command.begin(),command.end(),"-np");
        ASSERT_NE(ranks,command.end()); ASSERT_NE(std::next(ranks),command.end());
        EXPECT_EQ(*std::next(ranks),std::to_string(count));
        EXPECT_EQ(launch.numa_mapping->rankFileContents(),original_rankfile);
        EXPECT_EQ(std::count(original_rankfile.begin(),original_rankfile.end(),'\n'),count);
        EXPECT_NE(original_rankfile.find("rank 0=localhost slot=0\n"),std::string::npos);
        EXPECT_EQ(original_rankfile.find("+n"),std::string::npos);
        EXPECT_FALSE(contains(command,"--hostfile"));
        EXPECT_FALSE(contains(command,"--oversubscribe"));
        EXPECT_FALSE(contains(command,"--map-by"));
        EXPECT_FALSE(contains(command,"--cpu-set"));
    }
    MPILaunchConfig ordinary;
    EXPECT_FALSE(contains(buildCommand(ordinary,{}),"--host"));
    ordinary.hostfile = "/cluster/original hosts";
    const auto cluster = buildCommand(ordinary,{});
    EXPECT_FALSE(contains(cluster,"--host"));
    EXPECT_FALSE(contains(cluster,"--rankfile"));
    const auto hostfile = std::find(cluster.begin(),cluster.end(),"--hostfile");
    ASSERT_NE(hostfile,cluster.end()); ASSERT_NE(std::next(hostfile),cluster.end());
    EXPECT_EQ(*std::next(hostfile),ordinary.hostfile);
}

/** @test Every self-launched core-placed rank binds one place per allowed core, fastest first. */
TEST(Test__MPIBootstrap, CorePlacesFollowCapacityWithinTheRankMask)
{
    namespace fs = std::filesystem;
    const auto root = fs::temp_directory_path() /
        ("llaminar-core-places-" + std::to_string(::getpid()));
    const auto write = [&](int cpu, const std::string &leaf, const std::string &value) {
        fs::create_directories((root / ("cpu" + std::to_string(cpu)) / leaf).parent_path());
        std::ofstream(root / ("cpu" + std::to_string(cpu)) / leaf) << value << '\n';
    };
    // Recorded GB10 layout: no SMT, A725 0-4/10-14, X925 5-9/15-19.
    const std::vector<long> gb10 = {718, 718, 718, 718, 718, 997, 997, 997, 997, 997,
                                    731, 731, 731, 731, 731, 1017, 1017, 1017, 1017, 1024};
    std::vector<int> all;
    for (int cpu = 0; cpu < 20; ++cpu)
    {
        write(cpu, "cpu_capacity", std::to_string(gb10[cpu]));
        write(cpu, "topology/thread_siblings_list", std::to_string(cpu));
        all.push_back(cpu);
    }
    using Places = std::vector<std::vector<int>>;
    EXPECT_EQ(llaminar2::MPIBootstrapPhase::capacityOrderedCorePlaces(all, root.string()),
              (Places{{19}, {15}, {16}, {17}, {18}, {5}, {6}, {7}, {8}, {9},
                      {10}, {11}, {12}, {13}, {14}, {0}, {1}, {2}, {3}, {4}}));
    // A narrower rank mask is never widened.
    EXPECT_EQ(llaminar2::MPIBootstrapPhase::capacityOrderedCorePlaces({0, 1, 5, 6}, root.string()),
              (Places{{5}, {6}, {0}, {1}}));
    fs::remove_all(root);

    // SMT host without capacities: libgomp's core order (by first allowed
    // CPU), siblings kept together and restricted to the allowed mask.
    for (int core = 0; core < 4; ++core)
        for (const int cpu : {core, core + 4})
            write(cpu, "topology/thread_siblings_list", std::to_string(core) + "," + std::to_string(core + 4));
    EXPECT_EQ(llaminar2::MPIBootstrapPhase::capacityOrderedCorePlaces({7, 6, 5, 4, 3, 2, 1, 0}, root.string()),
              (Places{{0, 4}, {1, 5}, {2, 6}, {3, 7}}));
    EXPECT_EQ(llaminar2::MPIBootstrapPhase::capacityOrderedCorePlaces({2, 4, 5}, root.string()),
              (Places{{2}, {4}, {5}}));
    EXPECT_TRUE(llaminar2::MPIBootstrapPhase::capacityOrderedCorePlaces({}, root.string()).empty());
    fs::remove_all(root);
}

/** @test GB10 workers start on X925 cores; homogeneous hosts keep CPU order. */
TEST(Test__MPIBootstrap, ClusterWorkersOrderFastCoresFirst)
{
    namespace fs = std::filesystem;
    const auto root = fs::temp_directory_path() /
        ("llaminar-capacity-" + std::to_string(::getpid()));
    // Recorded GB10 capacities: two clusters of five A725 (718/731) then five
    // X925 (997..1024) cores.
    const std::vector<long> gb10 = {718, 718, 718, 718, 718, 997, 997, 997, 997, 997,
                                    731, 731, 731, 731, 731, 1017, 1017, 1017, 1017, 1024};
    for (int cpu = 0; cpu < static_cast<int>(gb10.size()); ++cpu)
    {
        fs::create_directories(root / ("cpu" + std::to_string(cpu)));
        std::ofstream(root / ("cpu" + std::to_string(cpu)) / "cpu_capacity") << gb10[cpu] << '\n';
    }
    std::vector<int> cpus(gb10.size());
    for (int cpu = 0; cpu < static_cast<int>(cpus.size()); ++cpu)
        cpus[static_cast<size_t>(cpu)] = cpu;
    EXPECT_EQ(llaminar2::MPIBootstrapPhase::capacityOrderedCpus(cpus, root.string()),
              (std::vector<int>{19, 15, 16, 17, 18, 5, 6, 7, 8, 9,
                                10, 11, 12, 13, 14, 0, 1, 2, 3, 4}));
    // Frequency is the fallback capacity; unknown topology keeps CPU order.
    fs::remove_all(root);
    for (int cpu = 0; cpu < 4; ++cpu)
    {
        fs::create_directories(root / ("cpu" + std::to_string(cpu)) / "cpufreq");
        std::ofstream(root / ("cpu" + std::to_string(cpu)) / "cpufreq" / "cpuinfo_max_freq")
            << (cpu % 2 ? 3900000 : 2808000) << '\n';
    }
    EXPECT_EQ(llaminar2::MPIBootstrapPhase::capacityOrderedCpus({0, 1, 2, 3}, root.string()),
              (std::vector<int>{1, 3, 0, 2}));
    EXPECT_EQ(llaminar2::MPIBootstrapPhase::capacityOrderedCpus({3, 2, 1, 0}, (root / "absent").string()),
              (std::vector<int>{0, 1, 2, 3}));
    fs::remove_all(root);
}
