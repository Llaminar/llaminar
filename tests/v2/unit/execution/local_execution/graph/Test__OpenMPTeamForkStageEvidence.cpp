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
            const char *stage_timing = nullptr, bool force_absent = false)
        {
            // EVERY enablement input is isolated: an ambient table, JSON or
            // CSV export, legacy profiling switch or GPU stage-timing family
            // would otherwise keep collection alive under the off-default.
            for (const char *name : {"LLAMINAR_PERF_STATS_SUMMARY",
                                     "LLAMINAR_PERF_STATS_FILTER",
                                     "LLAMINAR_PERF_STATS_TABLE",
                                     "LLAMINAR_PERF_STATS_JSON",
                                     "LLAMINAR_PERF_STATS_CSV",
                                     "LLAMINAR_PROFILING",
                                     "LLAMINAR_PERF_STATS_CPU_STAGE_TIMING",
                                     "LLAMINAR_PERF_STATS_GPU_STAGE_TIMING"})
                take(name);
            if (summary_filter)
            {
                ::setenv("LLAMINAR_PERF_STATS_SUMMARY", "1", 1);
                ::setenv("LLAMINAR_PERF_STATS_FILTER", summary_filter, 1);
            }
            else if (force_absent)
            {
                ::unsetenv("LLAMINAR_PERF_STATS_SUMMARY");
                ::unsetenv("LLAMINAR_PERF_STATS_FILTER");
            }
            if (stage_timing)
                ::setenv("LLAMINAR_PERF_STATS_CPU_STAGE_TIMING", stage_timing, 1);
            else if (force_absent)
            {
                // A forced-absent configuration must not leave ANY of the
                // enablement inputs ambient: clear all of them, not a subset.
                for (const char *name : {"LLAMINAR_PERF_STATS_SUMMARY",
                                         "LLAMINAR_PERF_STATS_FILTER",
                                         "LLAMINAR_PERF_STATS_TABLE",
                                         "LLAMINAR_PERF_STATS_JSON",
                                         "LLAMINAR_PERF_STATS_CSV",
                                         "LLAMINAR_PROFILING",
                                         "LLAMINAR_PERF_STATS_CPU_STAGE_TIMING",
                                         "LLAMINAR_PERF_STATS_GPU_STAGE_TIMING"})
                    ::unsetenv(name);
            }
            llaminar2::mutableDebugEnv().reload();
            llaminar2::PerfStatsCollector::reloadConfigurationFromEnvironment();
            llaminar2::PerfStatsCollector::reset();
        }
        ~ScopedForkEvidenceConfig()
        {
            // Restore the exact prior values, then reload both authorities
            // from the restored environment.
            for (const auto &entry : saved_)
            {
                if (entry.second)
                    ::setenv(entry.first.c_str(), entry.second->c_str(), 1);
                else
                    ::unsetenv(entry.first.c_str());
            }
            llaminar2::mutableDebugEnv().reload();
            llaminar2::PerfStatsCollector::reloadConfigurationFromEnvironment();
        }
        ScopedForkEvidenceConfig(const ScopedForkEvidenceConfig &) = delete;
        ScopedForkEvidenceConfig &operator=(const ScopedForkEvidenceConfig &) = delete;

    private:
        void take(const char *name)
        {
            if (const char *prior = ::getenv(name))
                saved_.emplace(name, std::string(prior));
            else
                saved_.emplace(name, std::nullopt);
        }
        std::map<std::string, std::optional<std::string>> saved_;
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
        const ScopedForkEvidenceConfig off_default("omp", nullptr, true);
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

        // The repeat contract is the collector's accumulator: the second
        // execution adds its deltas to the same keyed record, so the values
        // double (2→4, 5→10) and the observation count doubles (1→2). Changing
        // collector semantics to keep single-step values would break every
        // accumulating family; the test asserts the real contract.
        graph.reset();
        ASSERT_TRUE(executor.execute(graph, &context));
        for (const auto &record : llaminar2::PerfStatsCollector::snapshot({"omp"}))
            if (record.domain == "omp" &&
                record.name == llaminar::v2::kOpenMPTeamForkStageFamily &&
                record.tags.count("stage"))
            {
                const double expected = record.tags.at("stage") == "fork_stage_0" ? 4.0 : 10.0;
                EXPECT_EQ(record.value, expected)
                    << "two identical steps must accumulate " << expected;
                EXPECT_EQ(record.count, 2u);
            }
    }
}

/**
 * @brief The retained multi-device loop publishes the same exact cells.
 *
 * A CPU-only retained plan discriminates the retained publication seam from
 * the ordinary single-device loop: the same stages, executed through
 * executeMultiDevice, publish one record per CPU entry with the exact cell
 * values, and the publisher's GPU refusal holds for a GPU device string
 * without any GPU hardware.
 */
TEST(OpenMPTeamForkStageEvidence, RetainedLoopPublishesExactPerStageCells)
{
    const ScopedForkEvidenceConfig selected("omp", "1");
    llaminar2::CPUDeviceContext context{DeviceId::cpu()};
    DeviceGraphExecutor executor;
    ComputeGraph graph;
    graph.addNode("retained_stage_0", std::make_unique<ForkingStage>("retained_stage_0", DeviceId::cpu(), 2), DeviceId::cpu());
    graph.addNode("retained_stage_1", std::make_unique<ForkingStage>("retained_stage_1", DeviceId::cpu(), 5), DeviceId::cpu());
    graph.addDependency("retained_stage_1", "retained_stage_0");
    graph.buildFastSchedule();
    ASSERT_TRUE(executor.executeMultiDevice(graph, {{DeviceId::cpu(), &context}}));

    const auto records = llaminar2::PerfStatsCollector::snapshot({"omp"});
    const auto stage_value = [&records](const std::string &name) -> const PerfStatRecord *
    {
        for (const auto &record : records)
            if (record.domain == "omp" &&
                record.name == llaminar::v2::kOpenMPTeamForkStageFamily &&
                record.tags.count("stage") && record.tags.at("stage") == name)
                return &record;
        return nullptr;
    };
    const auto *first = stage_value("retained_stage_0");
    const auto *second = stage_value("retained_stage_1");
    ASSERT_NE(first, nullptr);
    ASSERT_NE(second, nullptr);
    EXPECT_EQ(first->value, 2.0);
    EXPECT_EQ(second->value, 5.0);

    // The publisher's own GPU refusal: a GPU device gains no record even with
    // a live owner scope and every gate selected (no GPU hardware needed).
    {
        llaminar::v2::OpenMPTeamForkScope scope;
        auto work = [] {};
        OMP_WORKSHARE_REGION(work);
        const auto before = records.size();
        llaminar::v2::publishOpenMPTeamForkStage(scope, "gpu_refused", "decode",
            DeviceId::cuda(0), 9u, "GEMM");
        EXPECT_EQ(llaminar2::PerfStatsCollector::snapshot({"omp"}).size(), before)
            << "a GPU device must never gain a fork record";
    }
}

/** @brief The retained multi-device plan publishes no fork record for GPU entries. */
TEST(OpenMPTeamForkStageEvidence, PublisherRefusesNonCpuDevicesAndUngatedCalls)
{
    // Control (a): EVERYTHING off. The publisher must refuse, and the gate
    // predicates prove the configuration actually is all-off before the call.
    {
        const ScopedForkEvidenceConfig all_off(nullptr, nullptr, true);
        EXPECT_FALSE(llaminar2::PerfStatsCollector::cpuStageTimingEnabled());
        EXPECT_FALSE(llaminar2::PerfStatsCollector::isRecordFamilyEnabled(
            "omp", llaminar::v2::kOpenMPTeamForkStageFamily));
        llaminar::v2::OpenMPTeamForkScope scope;
        auto work = [] {};
        OMP_WORKSHARE_REGION(work);
        llaminar::v2::publishOpenMPTeamForkStage(scope, "all_off_entry", "decode",
            DeviceId::cpu(), 1u, "GEMM");
        EXPECT_TRUE(llaminar2::PerfStatsCollector::snapshot({"omp"}).empty())
            << "an all-off configuration must publish nothing";
    }
    // Control (b): collection ON, the omp family selected, the CPU
    // stage-timing gate OFF. The predicates prove exactly that state before
    // the call, so the empty result is attributable to the gate alone - a
    // removed gate would flip this control to publishing.
    {
        const ScopedForkEvidenceConfig ungated("omp", nullptr, true);
        ::setenv("LLAMINAR_PERF_STATS_SUMMARY", "1", 1);
        ::setenv("LLAMINAR_PERF_STATS_FILTER", "omp", 1);
        mutableDebugEnv().reload();
        llaminar2::PerfStatsCollector::reloadConfigurationFromEnvironment();
        ASSERT_TRUE(llaminar2::PerfStatsCollector::isRecordFamilyEnabled(
            "omp", llaminar::v2::kOpenMPTeamForkStageFamily))
            << "the family must be selected for the gate refusal to mean anything";
        ASSERT_FALSE(llaminar2::PerfStatsCollector::cpuStageTimingEnabled())
            << "the stage gate must be the OFF input in this control";
        llaminar::v2::OpenMPTeamForkScope scope;
        auto work = [] {};
        OMP_WORKSHARE_REGION(work);
        llaminar::v2::publishOpenMPTeamForkStage(scope, "cpu_entry", "decode",
            DeviceId::cpu(), 1u, "GEMM");
        EXPECT_TRUE(llaminar2::PerfStatsCollector::snapshot({"omp"}).empty())
            << "the stage-timing gate alone must refuse the publication";
    }
    {
        const ScopedForkEvidenceConfig selected("omp", "1");
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
        llaminar::v2::publishOpenMPTeamForkSummary(scope, DeviceId::cuda(0),
            [] { return std::string("context"); }, false);
        llaminar::v2::publishOpenMPTeamForkStage(scope, "cpu_entry", "decode",
            DeviceId::cpu(), 7u, "GEMM");
        const auto records = llaminar2::PerfStatsCollector::snapshot({"omp"});
        // Exact keys: the earlier cpu_forks stage keeps its own 3-fork record
        // untouched; the refusal publishes nothing for the GPU device, and the
        // CPU-device publisher call lands exactly once with its own cell.
        const auto stage_record = [&records](const std::string &stage) -> const PerfStatRecord *
        {
            for (const auto &record : records)
                if (record.domain == "omp" &&
                    record.name == llaminar::v2::kOpenMPTeamForkStageFamily &&
                    record.tags.count("stage") && record.tags.at("stage") == stage)
                    return &record;
            return nullptr;
        };
        const auto *prior = stage_record("cpu_forks");
        ASSERT_NE(prior, nullptr);
        EXPECT_EQ(prior->value, 3.0);
        EXPECT_EQ(stage_record("gpu_entry"), nullptr)
            << "a GPU device must never gain a fork record";
        const auto *accepted = stage_record("cpu_entry");
        ASSERT_NE(accepted, nullptr);
        EXPECT_EQ(accepted->value, 1.0);
        EXPECT_EQ(accepted->device, "CPU")
            << "the canonical DeviceId::cpu().toString() spelling";
        EXPECT_TRUE(std::none_of(records.begin(), records.end(),
            [](const PerfStatRecord &record) { return record.device == "CUDA:0"; }));
    }
}
}

/** @brief Prior-present and prior-absent restoration are both exact. */
TEST(OpenMPTeamForkStageEvidence, EvidenceConfigRestoresExactPriorEnvironment)
{
    // Prior-present: ambient values survive the selected configuration.
    {
        const ScopedForkEvidenceConfig present(nullptr, nullptr, false);
        ::setenv("LLAMINAR_PERF_STATS_CPU_STAGE_TIMING", "0", 1);
        ::setenv("LLAMINAR_PERF_STATS_FILTER", "mtp", 1);
        llaminar2::mutableDebugEnv().reload();
        llaminar2::PerfStatsCollector::reloadConfigurationFromEnvironment();
        {
            const ScopedForkEvidenceConfig selected("omp", "1");
            EXPECT_EQ(std::getenv("LLAMINAR_PERF_STATS_FILTER"), std::string("omp"));
            EXPECT_EQ(std::getenv("LLAMINAR_PERF_STATS_CPU_STAGE_TIMING"), std::string("1"));
        }
        EXPECT_EQ(std::getenv("LLAMINAR_PERF_STATS_CPU_STAGE_TIMING"), std::string("0"));
        EXPECT_EQ(std::getenv("LLAMINAR_PERF_STATS_FILTER"), std::string("mtp"));
        ::unsetenv("LLAMINAR_PERF_STATS_CPU_STAGE_TIMING");
        ::unsetenv("LLAMINAR_PERF_STATS_FILTER");
        llaminar2::mutableDebugEnv().reload();
        llaminar2::PerfStatsCollector::reloadConfigurationFromEnvironment();
    }
    // Prior-absent: after a selected run, absence is restored as absence.
    {
        const ScopedForkEvidenceConfig absent(nullptr, nullptr, true);
        {
            const ScopedForkEvidenceConfig selected("omp", "1");
            EXPECT_EQ(std::getenv("LLAMINAR_PERF_STATS_FILTER"), std::string("omp"));
        }
        EXPECT_EQ(::getenv("LLAMINAR_PERF_STATS_FILTER"), nullptr);
        EXPECT_EQ(::getenv("LLAMINAR_PERF_STATS_SUMMARY"), nullptr);
        EXPECT_EQ(::getenv("LLAMINAR_PERF_STATS_CPU_STAGE_TIMING"), nullptr);
        EXPECT_FALSE(llaminar2::PerfStatsCollector::cpuStageTimingEnabled());
    }
}
