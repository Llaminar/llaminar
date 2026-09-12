/**
 * @file Test__MPIBootstrap.cpp
 * @brief Device-free launch policy and typed NUMA intent regressions.
 *
 * Test command construction with synthetic topology and validate malformed
 * placement before any hardware discovery, MPI launch, or model allocation.
 * Actual worker affinity belongs in the model-free CPU startup integration.
 */

#include "utils/MPIBootstrap.h"
#include "app/MPIBootstrapPhase.h"
#include "backends/ComputeBackend.h"
#include "config/ExecutionDomainDefinition.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <string>
#include <vector>

namespace
{
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
