/**
 * @file Test__HostEntryPreparation.cpp
 * @brief Device-free CPU preparation ordering across checked and retained execution.
 *
 * HostEntry retains cold metadata before any diagnostic or arithmetic consumer.
 * These tests use an empty-contract mock: no model, tensor kernels, device work,
 * publication transfer, or filesystem dump is needed to exercise real executor
 * boundaries. Repeated entry and request reset retain one original publication.
 */
#include <gtest/gtest.h>
#include "execution/local_execution/graph/DeviceGraphExecutor.h"
#include "execution/local_execution/device/DeviceContext.h"
#include "mocks/MockComputeStage.h"
#include "utils/DebugEnv.h"
#include <array>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

using namespace llaminar2;

namespace
{
    /** @brief Sentinel thrown at descriptor inspection before dump file setup. */
    class DiagnosticReached final : public std::runtime_error
    {
    public:
        /** @brief Construct the deliberate observation stop. */
        DiagnosticReached() : std::runtime_error("host diagnostic reached") {}
    };

    /** @brief Restore all executor debug switches touched by these device-free cases. */
    class ScopedHostEntryDebug final
    {
    public:
        /** @brief Select the direct retained route without changing process environment. */
        ScopedHostEntryDebug()
            : dump_(mutableDebugEnv().stage_dump), flags_{
                std::pair{&mutableDebugEnv().profile.enabled, mutableDebugEnv().profile.enabled},
                std::pair{&mutableDebugEnv().validation.validate_inputs, mutableDebugEnv().validation.validate_inputs},
                std::pair{&mutableDebugEnv().validation.validate_buffers, mutableDebugEnv().validation.validate_buffers},
                std::pair{&mutableDebugEnv().validation.validate_gpu_ptrs, mutableDebugEnv().validation.validate_gpu_ptrs},
                std::pair{&mutableDebugEnv().execution.trace_stages, mutableDebugEnv().execution.trace_stages},
                std::pair{&mutableDebugEnv().validation.sync_each_stage, mutableDebugEnv().validation.sync_each_stage}}
        {
            for (auto [flag, old] : flags_) { (void)old; *flag = false; }
            mutableDebugEnv().stage_dump.enabled = false;
        }
        /** @brief Restore configuration even when preparation or observation throws. */
        ~ScopedHostEntryDebug()
        {
            for (auto [flag, old] : flags_) *flag = old;
            mutableDebugEnv().stage_dump = dump_;
        }
    private:
        StageDumpConfig dump_;
        std::array<std::pair<bool *, bool>, 6> flags_;
    };

    /** @brief Stage that materializes one immutable binding and forbids late setup. */
    class HostEntryStage final : public llaminar2::testing::MockComputeStage
    {
    public:
        /** @brief Construct a CPU stage with independently selectable preparation policy. */
        explicit HostEntryStage(ComputeStageType type = ComputeStageType::GEMM,
            GraphLaunchPreparationPolicy policy = GraphLaunchPreparationPolicy::HostEntry)
            : MockComputeStage(type, "host_entry", DeviceId::cpu()), policy_(policy) {}

        /** @brief Declare the exact boundary the executor must invoke. */
        GraphLaunchPreparationPolicy graphLaunchPreparationPolicy() const override { return policy_; }

        /** @brief Retain one binding; repeated visits validate identity without republishing. */
        bool prepareGraphLaunch(IDeviceContext *ctx, void *stream) override
        {
            EXPECT_NE(ctx, nullptr);
            if (ctx) EXPECT_EQ(ctx->deviceId(), device());
            EXPECT_EQ(stream, nullptr);
            ++prepare_visits;
            events.push_back("prepare");
            if (!prepare_succeeds) return false;
            if (!prepared_) { prepared_ = true; ++publications; }
            return true;
        }

        /** @brief Require the cold binding before arithmetic can execute. */
        bool execute(IDeviceContext *ctx) override
        {
            if (policy_ == GraphLaunchPreparationPolicy::HostEntry && !prepared_)
                throw std::logic_error("execute cannot prepare missing metadata");
            events.push_back("execute");
            return MockComputeStage::execute(ctx);
        }

        /** @brief Prove descriptor inspection follows preparation, without reading live state. */
        StageDumpInfo buildDumpInfoImpl() const override
        {
            if (policy_ == GraphLaunchPreparationPolicy::HostEntry && !prepared_)
                throw std::logic_error("diagnostics cannot prepare missing metadata");
            events.push_back("observe");
            if (stop_at_diagnostic) throw DiagnosticReached();
            return {};
        }

        bool prepare_succeeds = true; ///< Inject an exact preparation failure before execution.
        bool stop_at_diagnostic = false; ///< Stop before the dumper creates any file or directory.
        int prepare_visits = 0; ///< Count lifecycle visits, not allocations/publications.
        int publications = 0; ///< One fake immutable metadata publication for the stage lifetime.
        mutable std::vector<std::string> events; ///< Observe ordering without real tensor work.
    private:
        GraphLaunchPreparationPolicy policy_;
        bool prepared_ = false; ///< Test binding state, never a copy of live request words.
    };

    /** @brief Own a real CPU executor and its minimal declarative graph fixture. */
    class HostEntryPreparation : public ::testing::Test
    {
    protected:
        /** @brief Append one independently owned CPU stage and return its observation handle. */
        HostEntryStage *add(ComputeStageType type = ComputeStageType::GEMM,
            GraphLaunchPreparationPolicy policy = GraphLaunchPreparationPolicy::HostEntry)
        {
            auto stage = std::make_unique<HostEntryStage>(type, policy);
            auto *result = stage.get();
            graph.addNode("host_entry", std::move(stage), DeviceId::cpu());
            graph.buildFastSchedule();
            return result;
        }
        ScopedHostEntryDebug debug_;
        CPUDeviceContext context{DeviceId::cpu()};
        DeviceGraphExecutor executor;
        ComputeGraph graph;
    };
}

TEST_F(HostEntryPreparation, FullFastMultiDeviceAndResetRetainOneMetadataPublication)
{
    auto *stage = add();
    ASSERT_TRUE(executor.execute(graph, &context));
    ASSERT_TRUE(executor.executeFastDecode(graph, &context));
    stage->resetSessionState();
    graph.reset();
    ASSERT_TRUE(executor.executeMultiDevice(graph, {{DeviceId::cpu(), &context}}));
    EXPECT_EQ(stage->prepare_visits, 3);
    EXPECT_EQ(stage->publications, 1);
    EXPECT_EQ(stage->executionCount(), 3);
    EXPECT_EQ(stage->events, (std::vector<std::string>{
        "prepare", "execute", "prepare", "execute", "prepare", "execute"}));
}

TEST_F(HostEntryPreparation, FastCollectiveBranchStillPreparesBeforeExecute)
{
    auto *stage = add(ComputeStageType::ALLREDUCE);
    const std::unordered_set<std::string> collective{"host_entry"};
    ASSERT_TRUE(executor.executeFastDecode(graph, &context, &collective));
    EXPECT_EQ(stage->prepare_visits, 1);
    EXPECT_EQ(stage->events, (std::vector<std::string>{"prepare", "execute"}));
}

TEST_F(HostEntryPreparation, InputDiagnosticDescriptorIsPreparedBeforeAnyExecution)
{
    auto *stage = add();
    stage->stop_at_diagnostic = true;
    auto &dump = mutableDebugEnv().stage_dump;
    dump.enabled = true;
    dump.dump_all_types = dump.dump_all_names = dump.dump_all_layers = dump.dump_all_iterations = true;
    dump.dump_rank = -1;
    // Descriptor lookup throws before beginDump(), so no diagnostic file is created.
    EXPECT_THROW(executor.execute(graph, &context), DiagnosticReached);
    EXPECT_EQ(stage->events, (std::vector<std::string>{"prepare", "observe"}));
    EXPECT_EQ(stage->executionCount(), 0);
    EXPECT_EQ(stage->publications, 1);
}

TEST_F(HostEntryPreparation, RetainedDirectAndDiagnosticRoutesEachPrepareOncePerEntry)
{
    auto *stage = add();
    DeviceGraphExecutor::RetainedMultiDeviceExecutionPlan plan;
    std::string error;
    ASSERT_TRUE(executor.prepareRetainedMultiDeviceExecutionPlan(
        graph, {{DeviceId::cpu(), &context}}, plan, &error)) << error;
    EXPECT_EQ(stage->prepare_visits, 0);
    ASSERT_TRUE(executor.executeRetainedMultiDevice(plan, &error)) << error;
    stage->resetSessionState();
    ASSERT_TRUE(executor.executeRetainedMultiDevice(plan, &error)) << error;
    mutableDebugEnv().validation.validate_inputs = true; // Select canonical runStage route.
    ASSERT_TRUE(executor.executeRetainedMultiDevice(plan, &error)) << error;
    EXPECT_EQ(stage->prepare_visits, 3);
    EXPECT_EQ(stage->publications, 1);
    EXPECT_EQ(stage->executionCount(), 3);
}

TEST_F(HostEntryPreparation, FailedPreparationStopsCheckedAndFastExecution)
{
    auto *stage = add();
    stage->prepare_succeeds = false;
    EXPECT_THROW(executor.execute(graph, &context), std::runtime_error);
    EXPECT_THROW(executor.executeFastDecode(graph, &context), std::runtime_error);
    EXPECT_EQ(stage->prepare_visits, 2);
    EXPECT_EQ(stage->publications, 0);
    EXPECT_EQ(stage->executionCount(), 0);
}

TEST_F(HostEntryPreparation, FailedPreparationStopsRetainedExecution)
{
    auto *stage = add();
    stage->prepare_succeeds = false;
    DeviceGraphExecutor::RetainedMultiDeviceExecutionPlan plan;
    std::string error;
    ASSERT_TRUE(executor.prepareRetainedMultiDeviceExecutionPlan(
        graph, {{DeviceId::cpu(), &context}}, plan, &error)) << error;
    EXPECT_FALSE(executor.executeRetainedMultiDevice(plan, &error));
    EXPECT_NE(error.find("metadata preparation failed"), std::string::npos);
    EXPECT_EQ(stage->prepare_visits, 1);
    EXPECT_EQ(stage->executionCount(), 0);
}

TEST_F(HostEntryPreparation, CaptureOnlyPreparationIsNeverInvokedByCPUEntry)
{
    auto *stage = add(ComputeStageType::GEMM, GraphLaunchPreparationPolicy::CaptureOnly);
    ASSERT_TRUE(executor.execute(graph, &context));
    ASSERT_TRUE(executor.executeFastDecode(graph, &context));
    DeviceGraphExecutor::RetainedMultiDeviceExecutionPlan plan;
    std::string error;
    ASSERT_TRUE(executor.prepareRetainedMultiDeviceExecutionPlan(
        graph, {{DeviceId::cpu(), &context}}, plan, &error)) << error;
    ASSERT_TRUE(executor.executeRetainedMultiDevice(plan, &error)) << error;
    EXPECT_EQ(stage->prepare_visits, 0);
    EXPECT_EQ(stage->executionCount(), 3);
}
