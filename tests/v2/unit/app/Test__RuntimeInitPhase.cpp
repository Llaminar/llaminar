/**
 * @file Test__RuntimeInitPhase.cpp
 * @brief Unit tests for runtime initialization branch behavior.
 */

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include "app/RuntimeInitPhase.h"
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
