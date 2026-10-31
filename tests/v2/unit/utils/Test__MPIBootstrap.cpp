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
#include "app/MPIBootstrapPhase.h"
#include "backends/ComputeBackend.h"
#include "config/ExecutionDomainDefinition.h"
#include "config/OrchestrationConfigParser.h"
#include "config/OrchestrationConfigDocument.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <cstdlib>
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

/** @test Existing callers keep the exact launch-flag consumption contract by default. */
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
        (std::vector<std::string>{"observer", "--workload", "work.json", "--output", "out", "--", "--device", "cpu:0"}));
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
