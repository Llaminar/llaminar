/**
 * @file Test__MPIBootstrap.cpp
 * @brief Device-free launch policy and typed NUMA intent regressions.
 *
 * Test command construction with synthetic topology and validate malformed
 * placement before any hardware discovery, MPI launch, or model allocation.
 * Actual worker affinity belongs in the model-free CPU startup integration.
 * Diagnostic export tests deliberately distinguish portable observation policy
 * from launcher-local device visibility and cloud authentication environment.
 */

#include "utils/MPIBootstrap.h"
#include "utils/DebugEnv.h"
#include "app/MPIBootstrapPhase.h"
#include "backends/ComputeBackend.h"
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
#include <string>
#include <vector>

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
            topology);
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
