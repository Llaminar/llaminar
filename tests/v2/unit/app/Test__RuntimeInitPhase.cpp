/**
 * @file Test__RuntimeInitPhase.cpp
 * @brief Unit tests for runtime initialization branch behavior.
 */

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include "app/RuntimeInitPhase.h"
#include "mocks/MockOrchestrationRunner.h"

#include <sstream>
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
