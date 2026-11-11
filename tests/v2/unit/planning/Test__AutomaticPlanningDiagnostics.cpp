/**
 * @file Test__AutomaticPlanningDiagnostics.cpp
 * @brief Device-free planner selection separates diagnostics from complete admission and ranking.
 *
 * Sparse test metadata and a synthetic observed-inventory fixture exercise the
 * production candidate compiler, physical memory authority and selector. Costs
 * are decision oracles, not hardware speed claims. Both logging policies retain
 * every candidate. No accelerator, sampled payload or MPI runtime is used.
 */
#include "../../utils/CPUExecutionTestGeometry.h"
#include "../../utils/PlanningGGUFFixture.h"
#include "../../utils/PlanningHostServiceFixture.h"
#include "config/OrchestrationConfigDocument.h"
#include "fixtures/backends/NativeOrderingEventGroupReplayAccess.h"
#include "planning/AutomaticOrchestrationPlanner.h"
#include "utils/Logger.h"

#include <gtest/gtest.h>

#include <array>
#include <cstddef>
#include <set>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

using namespace llaminar2;

namespace
{
    /** @brief Restore the process logging policy after any completed or failed selection. */
    class ScopedLogPolicy final
    {
    public:
        /**
         * @brief Select one diagnostics policy without changing planner inputs.
         * @param level Logging threshold for this synchronous device-free selection.
         */
        explicit ScopedLogPolicy(LogLevel level)
            : saved_(Logger::getInstance().getLogLevel())
        {
            Logger::getInstance().setLogLevel(level);
        }

        /** @brief Restore the caller's policy before another test can execute. */
        ~ScopedLogPolicy()
        {
            Logger::getInstance().setLogLevel(saved_);
        }

    private:
        LogLevel saved_;
    };

    /** @brief Capture emitted diagnostics without leaking stderr redirection on failure. */
    class DiagnosticCapture final
    {
    public:
        /** @brief Capture only this synchronous planner invocation. */
        DiagnosticCapture()
        {
            ::testing::internal::CaptureStderr();
        }

        /** @brief Restore stderr when selection throws before normal capture completion. */
        ~DiagnosticCapture()
        {
            if (active_)
                (void)::testing::internal::GetCapturedStderr();
        }

        /** @return Emitted diagnostics after restoring the process stderr descriptor. */
        std::string finish()
        {
            active_ = false;
            return ::testing::internal::GetCapturedStderr();
        }

    private:
        bool active_ = true;
    };

    /**
     * @brief Build a complete synthetic observed-inventory fixture for four GPU devices and CPU.
     * @param family CUDA or ROCm metadata family; no native backend is initialized.
     * @return One synthetic rank with explicit ownership, CPU geometry and directed peer observations.
     * @throws std::invalid_argument For a family without a GPU planning contract.
     */
    ClusterInventory syntheticObservedInventory(DeviceType family)
    {
        if (family != DeviceType::CUDA && family != DeviceType::ROCm)
            throw std::invalid_argument("Diagnostics fixture requires a synthetic GPU family");

        ClusterInventory result;
        result.world_size = 1;
        RankInventory rank;
        rank.rank = 0;
        rank.node_id = 0;
        rank.local_rank = 0;
        rank.hostname = "synthetic-diagnostics";
        rank.cpu.numa_node = 0;
        rank.cpu_cores = 4;
        rank.cpu.compute_units = 4;
        rank.cpu_worker_threads = 4;
        rank.cpu_execution = test::kSyntheticCPUExecutionGeometry;
        rank.cpu.memory_bytes = 512ULL << 30;
        rank.cpu.free_memory_bytes = rank.cpu.memory_bytes;
        rank.cpu_memory_bytes = rank.cpu.memory_bytes;
        for (int ordinal = 0; ordinal < 4; ++ordinal)
        {
            rank.gpus.push_back({.type = family, .local_device_id = ordinal,
                .memory_bytes = 512ULL << 30, .free_memory_bytes = 500ULL << 30,
                .compute_units = 64, .uuid = "synthetic-diagnostics-" + std::to_string(ordinal),
                .numa_node = 0});
        }
        // Explicit negative observations distinguish this fixture from missing
        // discovery data. They never replace the real native test inventory.
        auto &matrix = family == DeviceType::CUDA ? rank.p2p_cuda : rank.p2p_rocm;
        auto &degree = family == DeviceType::CUDA ? rank.p2p_cuda_count : rank.p2p_rocm_count;
        degree = 4;
        matrix.assign(16, false);
        result.ranks.push_back(std::move(rank));
        result.buildNodeAggregations();
        return result;
    }

    /** @brief Complete selection observations; no allocation or admission authority is duplicated. */
    struct SelectionObservation
    {
        std::vector<std::string> candidates;
        std::set<OrchestrationStrategy> strategies;
        std::string winner;
        std::string diagnostics;
        OrchestrationSelectionCounts counts;
        double cost = 0;
    };

    /**
     * @brief Execute complete production admission and ranking over synthetic metadata.
     * @param source One retained sparse test-model metadata source.
     * @param family GPU family in the synthetic observed-inventory fixture.
     * @param level Diagnostics threshold; candidate and cost inputs remain identical.
     * @return Every admitted candidate plus the winning configuration, counts, cost and diagnostics.
     * @throws std::exception For a rejected fixture or failed production planning contract.
     */
    SelectionObservation observeCompleteSelection(
        const PlanningModelSource &source, DeviceType family, LogLevel level)
    {
        ScopedLogPolicy policy(level);
        DiagnosticCapture capture;
        const auto inventory = syntheticObservedInventory(family);
        OrchestrationConfig request;
        request.model_path = source.path();
        request.max_seq_len = 128;
        request.prefill_max_bucket_size = 64;
        request.kv_cache_precision = "fp32";
        request.moe_rebalance.mode = MoERebalanceRuntimeMode::Off;
        request.automatic_planning.only_backends = std::vector{DeviceType::CPU, family};
        request.automatic_planning.only_strategies = std::vector{
            OrchestrationStrategy::SingleDevice,
            OrchestrationStrategy::TensorParallel,
            OrchestrationStrategy::ExpertOverlay};
        OrchestrationCandidateMemoryPolicy memory{
            .prefill_bucket_rows = {32, 64},
            .minimum_prefill_sequence_rows = 1,
            .maximum_cached_prefill_buckets = 8,
            .retained_host_services = test::planningHostServiceBudgets(1)};
        const std::array devices{
            DeviceId(family, 0), DeviceId(family, 1), DeviceId(family, 2), DeviceId(family, 3)};
        memory.native_ordering_events = test::NativeOrderingEventGroupReplayAccess::projection(devices);
        SelectionObservation result;
        const auto selected = AutomaticOrchestrationPlanner::select(
            request, source, inventory, memory, {64, 64},
            [&](const AdmittedOrchestrationCandidate &candidate,
                const OrchestrationPlanningWorkload &work)
            {
                EXPECT_TRUE(candidate.physicalAdmission().plan().fits());
                result.candidates.push_back(serializeOrchestrationConfig(candidate.config()));
                result.strategies.insert(candidate.strategy());
                return OrchestrationCostEstimate(work,
                    1.0 + candidate.devicePlans().size(), 0.01,
                    "synthetic diagnostics decision oracle");
            });
        result.winner = serializeOrchestrationConfig(selected.candidate().config());
        result.counts = selected.counts();
        result.cost = selected.cost().requestSeconds();
        result.diagnostics = capture.finish();
        return result;
    }
}

/** @test Complete planner work and decisions survive DEBUG policy without per-proposal geometry I/O. */
TEST(AutomaticPlanningDiagnostics, CompleteAdmissionAndSelectionRetainAllWorkWithoutDebugGeometryFlood)
{
    test::PlanningGGUFFixture fixture(true, false, GGUFTensorType::F32, 512);
    PlanningModelSource source(fixture.path());
    for (const auto family : {DeviceType::CUDA, DeviceType::ROCm})
    {
        SCOPED_TRACE(deviceTypeToString(family));
        const auto info = observeCompleteSelection(source, family, LogLevel::INFO);
        const auto debug = observeCompleteSelection(source, family, LogLevel::VERBOSITY_DEBUG);
        ASSERT_GT(info.counts.evaluated, 4u);
        EXPECT_EQ(info.counts.proposed, info.counts.evaluated + info.counts.capacity_rejected);
        EXPECT_EQ(debug.counts.proposed, info.counts.proposed);
        EXPECT_EQ(debug.counts.evaluated, info.counts.evaluated);
        EXPECT_EQ(debug.counts.capacity_rejected, info.counts.capacity_rejected);
        EXPECT_EQ(debug.candidates, info.candidates);
        EXPECT_EQ(debug.winner, info.winner);
        EXPECT_EQ(debug.cost, info.cost);
        EXPECT_EQ(debug.strategies, (std::set{
            OrchestrationStrategy::SingleDevice,
            OrchestrationStrategy::TensorParallel,
            OrchestrationStrategy::ExpertOverlay}));
        for (const auto phrase : {
            "Building execution plans for ",
            "Using named domains mode",
            "Resolved domain '",
            "[TensorParallelConfig] GQA replication:",
            "[MoEOverlayCapacity] fixed component BOM device=",
            "[MoEExpertOverlayRuntimePlan] Resolved overlay topology:",
            "MoE expert overlay runtime plan:"})
        {
            EXPECT_EQ(debug.diagnostics.find(phrase), std::string::npos) << phrase;
        }
    }
}
