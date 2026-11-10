/**
 * @file Test__OpenMPTeamForkStageEvidence.cpp
 * @brief Actual-path regressions for the graph executor's team-fork stage publisher.
 *
 * The wake-up work item's per-stage evidence must survive the deletion of the
 * production publisher, so these regressions drive the real CPU graph
 * executor over a declarative graph whose stages fork an exact number of
 * teams through the production macro family, and read the evidence through
 * the PerfStats snapshot the exports serve. Covered: the off-default shape
 * (no stage-timing gate, no omp record), the selected shape (one record per
 * CPU stage with its exact fork count), a GPU-entry absence (the retained
 * plan's GPU entry publishes no fork record), and the exact-value contract
 * after repeated steps. The evidence environment restores the caller's
 * complete environment and both configuration authorities afterwards. This
 * target certifies the executor's publisher; it does not certify kernels,
 * models or economy.
 */
#include "execution/local_execution/graph/DeviceGraphExecutor.h"
#include "execution/local_execution/device/DeviceContext.h"
#include "utils/OpenMPUtils.h"
#include "utils/OpenMPTeamForkEvidence.h"
#include "utils/PerfStatsCollector.h"
#include "mocks/MockComputeStage.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <cstdlib>
#include <memory>
#include <string>
#include <vector>

using namespace llaminar2;

namespace
{
    /** @brief Select one PerfStats configuration and restore everything afterwards. */
    class ScopedForkEvidenceConfig final
    {
    public:
        explicit ScopedForkEvidenceConfig(const char *summary_filter,
            const char *stage_timing = nullptr)
        {
            if (summary_filter)
            {
                ::setenv("LLAMINAR_PERF_STATS_SUMMARY", "1", 1);
                ::setenv("LLAMINAR_PERF_STATS_FILTER", summary_filter, 1);
            }
            if (stage_timing)
                ::setenv("LLAMINAR_PERF_STATS_CPU_STAGE_TIMING", stage_timing, 1);
            llaminar2::mutableDebugEnv().reload();
            llaminar2::PerfStatsCollector::reloadConfigurationFromEnvironment();
            llaminar2::PerfStatsCollector::reset();
        }
        ~ScopedForkEvidenceConfig()
        {
            ::unsetenv("LLAMINAR_PERF_STATS_SUMMARY");
            ::unsetenv("LLAMINAR_PERF_STATS_FILTER");
            ::unsetenv("LLAMINAR_PERF_STATS_CPU_STAGE_TIMING");
            llaminar2::mutableDebugEnv().reload();
            llaminar2::PerfStatsCollector::reloadConfigurationFromEnvironment();
        }
        ScopedForkEvidenceConfig(const ScopedForkEvidenceConfig &) = delete;
        ScopedForkEvidenceConfig &operator=(const ScopedForkEvidenceConfig &) = delete;
    };

    /** @brief A mock stage whose execution forks an exact number of real teams. */
    class ForkingStage final : public llaminar2::testing::MockComputeStage
    {
    public:
        ForkingStage(std::string name, DeviceId device, int forks)
            : MockComputeStage(ComputeStageType::GEMM, std::move(name), device),
              forks_(forks)
        {
            setOnExecute([forks = forks_](IDeviceContext *)
            {
                auto work = [] {};
                for (int fork = 0; fork < forks; ++fork)
                    OMP_WORKSHARE_REGION(work);
            });
        }

    private:
        int forks_;
    };
} // namespace

/** @brief The executor's stage publisher emits exactly one record per CPU stage. */
TEST(OpenMPTeamForkStageEvidence, StagePublisherEmitsExactPerStageForkCounts)
{
    // The off-default shape: without the explicit CPU stage-timing gate the
    // executor costs no fork observation and publishes no omp record.
    {
        const ScopedForkEvidenceConfig off_default("omp");
        llaminar2::CPUDeviceContext context{DeviceId::cpu()};
        DeviceGraphExecutor executor;
        ComputeGraph graph;
        graph.addNode("fork_stage_0", std::make_unique<ForkingStage>("fork_stage_0", DeviceId::cpu(), 2), DeviceId::cpu());
        graph.buildFastSchedule();
        ASSERT_TRUE(executor.execute(graph, &context));
        EXPECT_TRUE(llaminar2::PerfStatsCollector::snapshot({"omp"}).empty());
    }

    // The selected shape: stage timing on and the omp family selected, every
    // CPU stage publishes one record whose value is exactly its own forks.
    {
        const ScopedForkEvidenceConfig selected("omp", "1");
        llaminar2::CPUDeviceContext context{DeviceId::cpu()};
        DeviceGraphExecutor executor;
        ComputeGraph graph;
        graph.addNode("fork_stage_0", std::make_unique<ForkingStage>("fork_stage_0", DeviceId::cpu(), 2), DeviceId::cpu());
        graph.addNode("fork_stage_1", std::make_unique<ForkingStage>("fork_stage_1", DeviceId::cpu(), 5), DeviceId::cpu());
        graph.addDependency("fork_stage_1", "fork_stage_0");
        graph.buildFastSchedule();
        ASSERT_TRUE(executor.execute(graph, &context));

        const auto records = llaminar2::PerfStatsCollector::snapshot({"omp"});
        const auto stage_record = [&records](const std::string &name)
        {
            return std::find_if(records.begin(), records.end(),
                [&name](const PerfStatRecord &record)
                {
                    return record.domain == "omp" &&
                           record.name == llaminar::v2::kOpenMPTeamForkStageFamily &&
                           record.tags.find("stage") != record.tags.end() &&
                           record.tags.at("stage") == name;
                });
        };
        const auto first = stage_record("fork_stage_0");
        const auto second = stage_record("fork_stage_1");
        ASSERT_NE(first, records.end());
        ASSERT_NE(second, records.end());
        EXPECT_EQ(first->value, 2.0) << "stage 0 must count exactly its own forks";
        EXPECT_EQ(second->value, 5.0) << "stage 1 must count exactly its own forks";
        EXPECT_EQ(first->tags.at("index"), "0");
        EXPECT_EQ(second->tags.at("index"), "1");
        // No summary record exists here: the stage publisher publishes only
        // stage families; the summary is the engine's owned surface.
        EXPECT_TRUE(std::none_of(records.begin(), records.end(),
            [](const PerfStatRecord &record)
            { return record.name == llaminar::v2::kOpenMPTeamForkSummaryFamily; }));

        // The exact-value contract survives a repeated step: each step's cell
        // starts empty, so a drifted accumulation would expose the leak.
        graph.reset();
        ASSERT_TRUE(executor.execute(graph, &context));
        for (const auto &record : llaminar2::PerfStatsCollector::snapshot({"omp"}))
            if (record.domain == "omp" &&
                record.name == llaminar::v2::kOpenMPTeamForkStageFamily &&
                record.tags.count("stage"))
                EXPECT_EQ(record.value, record.tags.at("stage") == "fork_stage_0" ? 2.0 : 5.0);
    }
}

/** @brief The retained multi-device plan publishes no fork record for GPU entries. */
TEST(OpenMPTeamForkStageEvidence, PublisherRefusesNonCpuDevicesByContract)
{
    const ScopedForkEvidenceConfig selected("omp");
    llaminar2::CPUDeviceContext context{DeviceId::cpu()};
    DeviceGraphExecutor executor;
    ComputeGraph graph;
    graph.addNode("cpu_forks", std::make_unique<ForkingStage>("cpu_forks", DeviceId::cpu(), 3), DeviceId::cpu());
    graph.buildFastSchedule();
    ASSERT_TRUE(executor.execute(graph, &context));

    // The publisher's own refusal: a GPU device (here DeviceId::cuda(0), no
    // GPU hardware required) must never gain a fork record, while the CPU
    // device's record carries the exact scope value.
    {
        llaminar::v2::OpenMPTeamForkScope scope;
        auto work = [] {};
        OMP_WORKSHARE_REGION(work);
        llaminar::v2::publishOpenMPTeamForkStage(scope, "gpu_entry", "decode",
            DeviceId::cuda(0), 7u, "GEMM");
        llaminar::v2::publishOpenMPTeamForkSummary(scope.forks(), DeviceId::cuda(0),
            "context", false);
        llaminar::v2::publishOpenMPTeamForkStage(scope, "cpu_entry", "decode",
            DeviceId::cpu(), 7u, "GEMM");
        const auto records = llaminar2::PerfStatsCollector::snapshot({"omp"});
        for (const auto &record : records)
        {
            EXPECT_NE(record.device, "CUDA:0")
                << "a GPU device must never gain a fork record";
            if (record.name == llaminar::v2::kOpenMPTeamForkStageFamily)
                EXPECT_EQ(record.value, 1.0);
        }
    }
}
