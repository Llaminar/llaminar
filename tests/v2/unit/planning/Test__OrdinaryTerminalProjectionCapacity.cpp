/**
 * @file Test__OrdinaryTerminalProjectionCapacity.cpp
 * @brief Device-free declaration, bounds and existing runtime projection proofs.
 *
 * No model, backend or MPI is initialized. Tests exercise the real immutable
 * value and pre-compilation setup boundary, including disabled-but-retained
 * MTP capacity used by MainInference parity helpers. Byte pricing and actual
 * native capture remain in the existing planner and collective suites.
 */
#include "execution/config/OrdinaryTerminalProjectionCapacity.h"
#include "execution/factory/InferenceRunnerFactory.h"
#include "execution/local_execution/orchestrators/RankOrchestrator.h"
#include "execution/runner/OrchestrationRunner.h"
#include "execution/mpi_orchestration/IExecutionPlanBuilder.h"
#include <gtest/gtest.h>
#include <limits>

using namespace llaminar2;

/** @test Explicit rows are flattened demand, bounded by the actual activation family. */
TEST(OrdinaryTerminalProjectionCapacity, CheckedFlattenedDemandAndDefaultFinalRows)
{
    const auto ordinary = OrdinaryTerminalProjectionCapacity::selectedFinalRowsPerRequest();
    EXPECT_FALSE(ordinary.hasExplicitRows());
    EXPECT_EQ(ordinary.resolveForSequence(1, 512), 1);
    EXPECT_EQ(ordinary.resolveForSequence(4, 512), 4);
    const auto diagnostic = OrdinaryTerminalProjectionCapacity::projectedRows(16);
    EXPECT_EQ(diagnostic.resolveForSequence(4, 4), 16);
    EXPECT_EQ(diagnostic.resolveForSequence(4, 512), 16);
    EXPECT_THROW(diagnostic.resolveForSequence(1, 15), std::invalid_argument);
    EXPECT_THROW(OrdinaryTerminalProjectionCapacity::projectedRows(0), std::invalid_argument);
    EXPECT_THROW(OrdinaryTerminalProjectionCapacity::projectedRows(-1), std::invalid_argument);
    EXPECT_THROW(ordinary.resolveForSequence(0, 512), std::invalid_argument);
    EXPECT_THROW(ordinary.resolve(4, 3), std::invalid_argument);
    EXPECT_THROW(ordinary.resolveForSequence(2, std::numeric_limits<int>::max()), std::overflow_error);
}

/** @test True retained capacity is independent of enablement and transient forward role. */
TEST(OrdinaryTerminalProjectionCapacity, MainInferenceRetainsExactMTPUnion)
{
    MTPRuntimeConfig mtp;
    mtp.enabled = false;
    mtp.graph_capacity_draft_tokens = 3;
    const auto ordinary = OrdinaryTerminalProjectionCapacity{};
    const int retained = resolveMTPRetainedTargetQueryRows(mtp);
    ASSERT_GT(retained, 1);
    EXPECT_NO_THROW(ordinary.requirePublication(retained, 1, 512, retained));
    EXPECT_THROW(ordinary.requirePublication(retained + 1, 1, 512, retained), std::invalid_argument);
    mtp.graph_capacity_draft_tokens = 0;
    ASSERT_EQ(resolveMTPRetainedTargetQueryRows(mtp), 0);
    EXPECT_THROW(ordinary.requirePublication(2, 1, 512, 0), std::invalid_argument);
    const auto diagnostic = OrdinaryTerminalProjectionCapacity::projectedRows(16);
    EXPECT_NO_THROW(diagnostic.requirePublication(16, 1, 512, retained));
    EXPECT_THROW(diagnostic.requirePublication(17, 1, 512, retained), std::invalid_argument);
}

/** @test Both participant adapters preserve one authored declaration without enabling MTP. */
TEST(OrdinaryTerminalProjectionCapacity, RuntimeProjectionPreservesDeclaration)
{
    RuntimeConfig runtime;
    runtime.batch_size = 2;
    runtime.resident_graph_rows = 32;
    runtime.ordinary_terminal_projection = OrdinaryTerminalProjectionCapacity::projectedRows(16);
    const auto rank = RankOrchestrator::Config::fromRuntime(runtime);
    const auto device = InferenceRunnerConfig::fromRuntime(runtime);
    EXPECT_EQ(rank.ordinary_terminal_projection, runtime.ordinary_terminal_projection);
    EXPECT_EQ(device.ordinary_terminal_projection, runtime.ordinary_terminal_projection);
    EXPECT_FALSE(rank.mtp.enabled);
    EXPECT_FALSE(device.mtp.enabled);
    EXPECT_EQ(rank.mtp.graph_capacity_draft_tokens, 0);
    EXPECT_EQ(device.mtp.graph_capacity_draft_tokens, 0);
}

/** @test Late output selection cannot rewrite an already compiled capacity declaration. */
TEST(OrdinaryTerminalProjectionCapacity, SetupRejectsLateAndPreservesPrecompiledDemand)
{
    OrchestrationConfig config;
    OrchestrationRunner setup(config, std::unique_ptr<IExecutionPlanBuilder>{});
    EXPECT_TRUE(setup.setOrdinaryTerminalProjectionCapacity(
        OrdinaryTerminalProjectionCapacity::projectedRows(16)));
    RankExecutionPlan plan;
    plan.runtime.ordinary_terminal_projection = OrdinaryTerminalProjectionCapacity::projectedRows(7);
    OrchestrationRunner compiled(config, plan);
    EXPECT_FALSE(compiled.setOrdinaryTerminalProjectionCapacity(
        OrdinaryTerminalProjectionCapacity::projectedRows(16)));
    EXPECT_EQ(compiled.executionPlan().runtime.ordinary_terminal_projection, plan.runtime.ordinary_terminal_projection);
    EXPECT_NE(compiled.lastError().find("before plan compilation"), std::string::npos);
}
