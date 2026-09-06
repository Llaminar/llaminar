/**
 * @file Test__RuntimeInitPhase.cpp
 * @brief Unit tests for runtime initialization branch behavior.
 *
 * Bootstrap intent checks are pure typed-config tests: no vendor enumeration,
 * model loading, environment mutation, or accelerator occupancy is permitted.
 */

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include "app/RuntimeInitPhase.h"
#include "app/MPIBootstrapPhase.h"
#include "utils/NUMATopology.h"
#include "mocks/MockOrchestrationRunner.h"

#include <sstream>
#include <stdexcept>
#include <string>

using namespace llaminar2;
using namespace llaminar2::test;
using ::testing::NiceMock;
using ::testing::Return;
using ::testing::ReturnRef;

TEST(Test__RuntimeInitPhase, DryRunPreflightInitializesRunnerAndPrintsResolvedPlanOnRankZero)
{
    OrchestrationConfig config = OrchestrationConfig::defaults();
    config.dry_run = true;
    config.model_path = "models/example.gguf";

    NiceMock<MockOrchestrationRunner> runner;
    std::ostringstream out;

    EXPECT_CALL(runner, initializeForDryRun()).WillOnce(Return(true));
    EXPECT_CALL(runner, config()).WillOnce(ReturnRef(config));
    EXPECT_CALL(runner, shutdown()).Times(1);

    EXPECT_TRUE(RuntimeInitPhase::runDryRunPreflight(config, runner, 0, out));
    EXPECT_TRUE(config.dry_run);

    const std::string output = out.str();
    EXPECT_NE(output.find("=== Resolved Orchestration Plan ==="), std::string::npos);
    EXPECT_NE(output.find("dry_run: true"), std::string::npos);
}

TEST(Test__RuntimeInitPhase, DryRunPreflightDoesNotPrintResolvedPlanOnWorkerRanks)
{
    OrchestrationConfig config = OrchestrationConfig::defaults();
    config.dry_run = true;
    config.model_path = "models/example.gguf";

    NiceMock<MockOrchestrationRunner> runner;
    std::ostringstream out;

    EXPECT_CALL(runner, initializeForDryRun()).WillOnce(Return(true));
    EXPECT_CALL(runner, config()).Times(0);
    EXPECT_CALL(runner, shutdown()).Times(1);

    EXPECT_TRUE(RuntimeInitPhase::runDryRunPreflight(config, runner, 1, out));
    EXPECT_TRUE(config.dry_run);
    EXPECT_TRUE(out.str().empty());
}

TEST(Test__RuntimeInitPhase, DryRunPreflightFailureClearsDryRunFlagForNonzeroExit)
{
    OrchestrationConfig config = OrchestrationConfig::defaults();
    config.dry_run = true;
    config.model_path = "models/example.gguf";

    NiceMock<MockOrchestrationRunner> runner;
    std::ostringstream out;
    const std::string error = "model validation failed";

    EXPECT_CALL(runner, initializeForDryRun()).WillOnce(Return(false));
    EXPECT_CALL(runner, lastError()).WillOnce(ReturnRef(error));
    EXPECT_CALL(runner, shutdown()).Times(1);

    EXPECT_FALSE(RuntimeInitPhase::runDryRunPreflight(config, runner, 0, out));
    EXPECT_FALSE(config.dry_run);
    EXPECT_TRUE(out.str().empty());
}

TEST(Test__RuntimeInitPhase, CPUBackendPlacementUsesExplicitRankDeviceMap)
{
    OrchestrationConfig config;
    config.device_map.emplace_back(1, GlobalDeviceAddress::cpu(1));
    config.device_map_numa_explicit.emplace_back(1, true);
    const NUMAInfo numa_info{
        .local_numa_node = 1,
        .total_numa_nodes = 2,
        .detection_succeeded = true,
        .detection_method = "test"};

    EXPECT_EQ(
        RuntimeInitPhase::resolveCPUBackendNUMANode(config, 1, 2, numa_info),
        1);
}

TEST(Test__RuntimeInitPhase, CPUBackendPlacementRejectsAffinityMismatch)
{
    OrchestrationConfig config;
    config.device_map.emplace_back(1, GlobalDeviceAddress::cpu(1));
    config.device_map_numa_explicit.emplace_back(1, true);
    const NUMAInfo numa_info{
        .local_numa_node = 0,
        .total_numa_nodes = 2,
        .detection_succeeded = true,
        .detection_method = "test"};

    EXPECT_THROW(
        RuntimeInitPhase::resolveCPUBackendNUMANode(config, 1, 2, numa_info),
        std::runtime_error);
}

TEST(Test__RuntimeInitPhase, CPUBackendPlacementUsesDetectedNodeForMultiRankRuntime)
{
    OrchestrationConfig config;
    const NUMAInfo numa_info{
        .local_numa_node = 1,
        .total_numa_nodes = 2,
        .detection_succeeded = true,
        .detection_method = "test"};

    EXPECT_EQ(
        RuntimeInitPhase::resolveCPUBackendNUMANode(config, 1, 2, numa_info),
        1);
}

TEST(Test__RuntimeInitPhase, CPUBackendPlacementKeepsSingleProcessAggregateExplicit)
{
    OrchestrationConfig config;
    const NUMAInfo numa_info{
        .local_numa_node = 0,
        .total_numa_nodes = 2,
        .detection_succeeded = true,
        .detection_method = "test"};

    EXPECT_EQ(
        RuntimeInitPhase::resolveCPUBackendNUMANode(config, 0, 1, numa_info),
        -1);
}

TEST(Test__RuntimeInitPhase, RankAgnosticOverlayGpuRequiresHostWideVisibility)
{
    OrchestrationConfig config = OrchestrationConfig::defaults();
    config.moe_routed_expert_plan =
        std::make_shared<MoERoutedExpertPlacementPlan>();
    config.moe_routed_expert_plan->enabled = true;
    RoutedExpertDomain domain;
    domain.name = "accelerator-tier";
    domain.scope = ExecutionDomainScope::RANK_LOCAL;
    domain.participants = {
        GlobalDeviceAddress::rocm(0),
        GlobalDeviceAddress::rocm(1),
    };
    config.moe_routed_expert_plan->domains.push_back(std::move(domain));

    EXPECT_TRUE(
        RuntimeInitPhase::requiresHostWideAcceleratorVisibility(config, 0));
}

TEST(Test__RuntimeInitPhase, CpuOnlyOverlayRetainsRankLocalVisibility)
{
    OrchestrationConfig config = OrchestrationConfig::defaults();
    config.moe_routed_expert_plan =
        std::make_shared<MoERoutedExpertPlacementPlan>();
    config.moe_routed_expert_plan->enabled = true;
    RoutedExpertDomain domain;
    domain.name = "cpu-tier";
    domain.scope = ExecutionDomainScope::SINGLE;
    domain.participants = {GlobalDeviceAddress::cpu(1)};
    config.moe_routed_expert_plan->domains.push_back(std::move(domain));

    EXPECT_FALSE(
        RuntimeInitPhase::requiresHostWideAcceleratorVisibility(config, 0));
}

TEST(Test__RuntimeInitPhase, BootstrapDoesNotConfuseMpiWithGpuOrCpuIntent)
{
    OrchestrationConfig config;
    config.default_backend = CollectiveBackendType::MPI;
    config.mpi_procs = 2;
    EXPECT_EQ(MPIBootstrapPhase::classifyDeviceIntent(config),
              BootstrapDeviceIntent::Automatic);
    config.domain_definitions.push_back(DomainDefinition::parse(
        "arbitrary_name=localhost:0:cpu:0,localhost:1:cpu:0;scope=node_local;backend=mpi"));
    EXPECT_EQ(MPIBootstrapPhase::classifyDeviceIntent(config),
              BootstrapDeviceIntent::CpuOnly);
}

TEST(Test__RuntimeInitPhase, BootstrapCpuIntentCoversDeviceMapTpAndTree)
{
    OrchestrationConfig shorthand;
    shorthand.cpu_global_tp_all_local = true;
    EXPECT_EQ(MPIBootstrapPhase::classifyDeviceIntent(shorthand),
              BootstrapDeviceIntent::CpuOnly);

    OrchestrationConfig single;
    single.device_for_this_rank = GlobalDeviceAddress::cpu(0);
    EXPECT_EQ(MPIBootstrapPhase::classifyDeviceIntent(single),
              BootstrapDeviceIntent::CpuOnly);

    OrchestrationConfig mapped;
    mapped.device_map = {{0, GlobalDeviceAddress::cpu(0)},
                         {1, GlobalDeviceAddress::cpu(1)}};
    EXPECT_EQ(MPIBootstrapPhase::classifyDeviceIntent(mapped),
              BootstrapDeviceIntent::CpuOnly);

    OrchestrationConfig tp;
    tp.tp_degree = 2;
    tp.tp_devices = {GlobalDeviceAddress::cpu(0), GlobalDeviceAddress::cpu(1)};
    EXPECT_EQ(MPIBootstrapPhase::classifyDeviceIntent(tp),
              BootstrapDeviceIntent::CpuOnly);

    OrchestrationConfig tree;
    tree.topology_tree.emplace();
    tree.topology_tree->root.device = GlobalDeviceAddress::cpu(1);
    EXPECT_EQ(MPIBootstrapPhase::classifyDeviceIntent(tree),
              BootstrapDeviceIntent::CpuOnly);
    tree.topology_tree.reset();
    tree.topology_string = "unresolved topology";
    tree.device_for_this_rank = GlobalDeviceAddress::cpu(0);
    EXPECT_EQ(MPIBootstrapPhase::classifyDeviceIntent(tree),
              BootstrapDeviceIntent::Automatic);
}

TEST(Test__RuntimeInitPhase, BootstrapGpuExpertsOverrideCpuContinuationSymmetrically)
{
    for (const auto gpu : {GlobalDeviceAddress::cuda(0),
                           GlobalDeviceAddress::rocm(0)})
    {
        OrchestrationConfig config;
        config.device_for_this_rank = GlobalDeviceAddress::cpu(0);
        config.moe_routed_expert_plan =
            std::make_shared<MoERoutedExpertPlacementPlan>();
        auto &overlay = *config.moe_routed_expert_plan;
        overlay.enabled = true;
        RoutedExpertDomain domain;
        domain.name = "typed_participants_not_tier_names";
        domain.participants = {GlobalDeviceAddress::cpu(1)};
        overlay.domains.push_back(domain);
        EXPECT_EQ(MPIBootstrapPhase::classifyDeviceIntent(config),
                  BootstrapDeviceIntent::CpuOnly);
        overlay.domains.front().participants.push_back(gpu);
        EXPECT_EQ(MPIBootstrapPhase::classifyDeviceIntent(config),
                  BootstrapDeviceIntent::Accelerator);

        overlay.domains.front().participants.pop_back();
        ExecutionDomainDefinition dense;
        dense.participants = {gpu};
        overlay.dense_domains.push_back(std::move(dense));
        EXPECT_EQ(MPIBootstrapPhase::classifyDeviceIntent(config),
                  BootstrapDeviceIntent::Accelerator);
    }
}

TEST(Test__RuntimeInitPhase, BootstrapGpuDeclarationsAreNotOverriddenByCpuShorthand)
{
    for (const auto gpu : {GlobalDeviceAddress::cuda(0),
                           GlobalDeviceAddress::rocm(0)})
    {
        OrchestrationConfig mapped;
        mapped.cpu_global_tp_all_local = true;
        mapped.device_map = {{1, gpu}};
        EXPECT_EQ(MPIBootstrapPhase::classifyDeviceIntent(mapped),
                  BootstrapDeviceIntent::Accelerator);
        mapped.device_map.clear();
        mapped.tp_devices = {gpu};
        EXPECT_EQ(MPIBootstrapPhase::classifyDeviceIntent(mapped),
                  BootstrapDeviceIntent::Accelerator);
        mapped.tp_devices.clear();
        mapped.topology_tree.emplace();
        mapped.topology_tree->root.device = gpu;
        EXPECT_EQ(MPIBootstrapPhase::classifyDeviceIntent(mapped),
                  BootstrapDeviceIntent::Accelerator);
    }
}
