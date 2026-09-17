/**
 * @file Test__AutomaticOrchestrationPlanner.cpp
 * @brief Device-free proof that complete admission precedes cost-ranked selection.
 *
 * Costs below are deliberately synthetic decision oracles, not speed claims.
 * The model directory, candidate compiler, memory authority and apply codec
 * are real production components. No tensor payload or accelerator is used.
 */
#include "planning/AutomaticOrchestrationPlanner.h"
#include "planning/AutomaticOrchestrationCandidates.h"
#include "planning/AutomaticPlanningStartup.h"
#include "planning/PhysicalMemoryCapacityExhausted.h"
#include "config/OrchestrationConfigDocument.h"
#include "../../utils/PlanningGGUFFixture.h"
#include <gtest/gtest.h>
#include <algorithm>
#include <cmath>
#include <limits>
#include <set>

using namespace llaminar2;

namespace
{
    /** @return Two choices whose larger resource is deliberately first in discovery. */
    ClusterInventory inventory(DeviceType backend)
    {
        ClusterInventory result;
        result.world_size = backend == DeviceType::CPU ? 2 : 1;
        for (int rank = 0; rank < result.world_size; ++rank)
        {
            RankInventory observed;
            observed.rank = rank;
            observed.node_id = rank;
            observed.hostname = "host-" + std::to_string(rank);
            observed.cpu.numa_node = 3 + rank;
            observed.cpu.memory_bytes = rank == 0 ? 64ull << 30 : 16ull << 30;
            observed.cpu.free_memory_bytes = observed.cpu.memory_bytes;
            observed.cpu_cores = observed.cpu.compute_units = rank == 0 ? 32 : 8;
            observed.cpu_worker_threads = rank == 0 ? 4 : 2;
            observed.numa_nodes = 1;
            if (backend != DeviceType::CPU)
            {
                observed.gpus.push_back({.type = backend, .local_device_id = 2,
                    .memory_bytes = 64ull << 30, .free_memory_bytes = 48ull << 30,
                    .compute_units = 128, .uuid = "larger", .numa_node = 3});
                observed.gpus.push_back({.type = backend, .local_device_id = 9,
                    .memory_bytes = 16ull << 30, .free_memory_bytes = 12ull << 30,
                    .compute_units = 32, .uuid = "smaller", .numa_node = 3});
            }
            result.ranks.push_back(std::move(observed));
        }
        result.buildNodeAggregations();
        return result;
    }

    /** @return Policy-only request; selection must preserve all non-placement fields. */
    OrchestrationConfig request(const PlanningModelSource &source, DeviceType backend)
    {
        OrchestrationConfig config;
        config.model_path = source.path();
        config.max_seq_len = 1024;
        config.automatic_planning.only_backends = std::vector{backend};
        config.automatic_planning.only_strategies = std::vector{OrchestrationStrategy::SingleDevice};
        return config;
    }

    /** @return Small capture ladder using the existing production upload policy. */
    OrchestrationCandidateMemoryPolicy memory()
    {
        return {.prefill_bucket_rows = {32, 64}, .minimum_prefill_sequence_rows = 1,
            .maximum_cached_prefill_buckets = 8};
    }

    /** @return Identity of the deliberately smaller, later-observed choice. */
    bool smaller(const AdmittedOrchestrationCandidate &candidate)
    {
        const auto &address = *candidate.config().device_for_this_rank;
        return address.isCPU() ? address.numa_node == 4 : address.device_ordinal == 9;
    }
}

TEST(AutomaticOrchestrationPlanner, RejectsMissingAndNonfiniteCostEvidence)
{
    EXPECT_THROW(OrchestrationPlanningWorkload(0, 1), std::invalid_argument);
    EXPECT_THROW(OrchestrationPlanningWorkload(1, -1), std::invalid_argument);
    const OrchestrationPlanningWorkload work(64, 128);
    for (double bad : {0.0, -1.0, std::numeric_limits<double>::infinity(),
                      std::numeric_limits<double>::quiet_NaN()})
    {
        EXPECT_THROW(OrchestrationCostEstimate(work, bad, 0.01, "fixture"), std::invalid_argument);
        EXPECT_THROW(OrchestrationCostEstimate(work, 1.0, bad, "fixture"), std::invalid_argument);
    }
    EXPECT_THROW(OrchestrationCostEstimate(work, 1.0, 0.01, " \t\n"), std::invalid_argument);
    EXPECT_THROW(OrchestrationCostEstimate(work, 1.0, std::numeric_limits<double>::max(), "fixture"), std::invalid_argument);
    const OrchestrationCostEstimate cost(work, 1.0, 0.01, "synthetic decision oracle");
    EXPECT_DOUBLE_EQ(cost.requestSeconds(), 2.28);
    EXPECT_EQ(cost.workload(), work);
}

TEST(AutomaticOrchestrationPlanner, StartupSearchSharesPolicyAndPreservesDiscoveryWithoutMutatingRequest)
{
    test::PlanningGGUFFixture file;
    PlanningModelSource source(file.path());
    const auto observed = inventory(DeviceType::CUDA);
    auto config = request(source, DeviceType::CUDA);
    config.max_seq_len = 128;
    config.prefill_max_bucket_size = 64;
    config.kv_cache_precision = "fp32";
    const auto original = serializeOrchestrationConfig(config);
    std::size_t calls = 0;
    const auto evaluate = [&](const auto &candidate, const auto &work) {
            ++calls;
            EXPECT_EQ(work, OrchestrationPlanningWorkload(64, 64));
            EXPECT_TRUE(candidate.physicalAdmission().plan().fits());
            return OrchestrationCostEstimate(work, smaller(candidate) ? 1 : 2, .01, "startup decision oracle");
        };
    size_t preparations = 0;
    const auto startup = AutomaticPlanningStartup::run(config, observed, {},
        [&](const AutomaticPlanningPreparation &context)
            -> std::optional<AutomaticOrchestrationPlanner::Evaluate> {
            ++preparations;
            EXPECT_EQ(calls, 0u); // No candidate may be priced before evidence is complete.
            EXPECT_TRUE(context.isRoot());
            EXPECT_FALSE(context.mpi());
            EXPECT_EQ(context.rootSource().path(), config.model_path);
            EXPECT_EQ(context.metadata().serialize(), source.metadata().serialize());
            EXPECT_EQ(context.workload(), OrchestrationPlanningWorkload(64, 64));
            EXPECT_EQ(context.request().mpi_procs, observed.world_size);
            return evaluate;
        });
    ASSERT_TRUE(startup.root_selection);
    const auto &winner = *startup.root_selection;
    EXPECT_EQ(preparations, 1u);
    EXPECT_TRUE(smaller(winner.candidate()));
    EXPECT_EQ(calls, 2u);
    EXPECT_EQ(winner.candidate().config().mpi_procs, observed.world_size);
    EXPECT_EQ(winner.candidate().membership().discoveryRanks(), (std::vector<int>{0}));
    EXPECT_EQ(serializeOrchestrationConfig(startup.applied), serializeOrchestrationConfig(winner.candidate().config()));
    EXPECT_EQ(winner.candidate().config().kv_cache_precision, "fp32");
    EXPECT_EQ(winner.candidate().config().prefill_max_bucket_size, 64);
    EXPECT_EQ(serializeOrchestrationConfig(config), original);
    EXPECT_THROW(AutomaticPlanningStartup::run(config, observed, {}, {}), std::runtime_error);
    for (const int invalid : {-1, 0, 1})
    {
        config.max_seq_len = invalid;
        EXPECT_THROW(AutomaticPlanningStartup::run(config, observed, {}), std::runtime_error);
    }
}

TEST(AutomaticOrchestrationPlanner, SmallerFasterLaterDeviceWinsOnEveryBackend)
{
    test::PlanningGGUFFixture file;
    PlanningModelSource source(file.path());
    for (const auto backend : {DeviceType::CPU, DeviceType::CUDA, DeviceType::ROCm})
    {
        SCOPED_TRACE(deviceTypeToString(backend));
        const auto config = request(source, backend);
        const auto original = serializeOrchestrationConfig(config);
        size_t calls = 0;
        const auto selected = AutomaticOrchestrationPlanner::select(config, source, inventory(backend), memory(), {64, 128},
            [&](const auto &admitted, const auto &work) {
                ++calls;
                EXPECT_TRUE(admitted.physicalAdmission().plan().fits());
                EXPECT_TRUE(admitted.config().execution_rank_selection);
                return OrchestrationCostEstimate(work, smaller(admitted) ? 0.1 : 0.2, 0.01, "synthetic decision oracle");
            });
        EXPECT_TRUE(smaller(selected.candidate()));
        EXPECT_EQ(calls, 2u);
        EXPECT_EQ(selected.counts().proposed, calls);
        EXPECT_EQ(selected.counts().evaluated, calls);
        EXPECT_EQ(selected.counts().capacity_rejected, 0u);
        EXPECT_EQ(serializeOrchestrationConfig(config), original);
        const auto saved = serializeOrchestrationConfig(selected.candidate().config());
        const auto applied = deserializeOrchestrationConfig(saved);
        EXPECT_EQ(serializeOrchestrationConfig(applied), saved);
        EXPECT_EQ(*applied.execution_rank_selection, selected.candidate().membership().selection());
    }
}

TEST(AutomaticOrchestrationPlanner, RequestHorizonCanMakeTensorParallelFasterThanSingle)
{
    test::PlanningGGUFFixture file;
    PlanningModelSource source(file.path());
    for (const auto backend : {DeviceType::CUDA, DeviceType::ROCm})
    {
        auto config = request(source, backend);
        config.automatic_planning.only_strategies =
            std::vector{OrchestrationStrategy::SingleDevice, OrchestrationStrategy::TensorParallel};
        const auto cost = [](const auto &candidate, const auto &work) {
            const bool tp = candidate.strategy() == OrchestrationStrategy::TensorParallel;
            return OrchestrationCostEstimate(work, tp ? 1.0 : 0.8, tp ? 0.02 : 0.03, "synthetic phase costs");
        };
        config.automatic_planning.workload = OrchestrationPlanningWorkload(64, 1);
        const auto short_request = AutomaticOrchestrationPlanner::select(config, source, inventory(backend),
            OrchestrationCandidateMemoryPolicy::fromStartup(), *config.automatic_planning.workload, cost);
        config.automatic_planning.workload = OrchestrationPlanningWorkload(64, 128);
        const auto long_request = AutomaticOrchestrationPlanner::select(config, source, inventory(backend),
            OrchestrationCandidateMemoryPolicy::fromStartup(), *config.automatic_planning.workload, cost);
        EXPECT_EQ(short_request.cost().workload(), OrchestrationPlanningWorkload(64, 1));
        EXPECT_EQ(long_request.cost().workload(), *config.automatic_planning.workload);
        EXPECT_FALSE(long_request.candidate().config().automatic_planning.specified());
        EXPECT_THROW(AutomaticOrchestrationPlanner::select(config, source, inventory(backend), memory(),
            {64, 1}, cost), std::invalid_argument);
        EXPECT_EQ(short_request.candidate().strategy(), OrchestrationStrategy::SingleDevice);
        EXPECT_EQ(long_request.candidate().strategy(), OrchestrationStrategy::TensorParallel);
        EXPECT_EQ(short_request.counts().evaluated, 3u);
        EXPECT_EQ(long_request.counts().evaluated, 3u);
    }
}

TEST(AutomaticOrchestrationPlanner, EqualCostsPreferSingleDeviceOverOneRankMultiDevice)
{
    test::PlanningGGUFFixture file;
    PlanningModelSource source(file.path());
    for (const auto backend : {DeviceType::CUDA, DeviceType::ROCm})
    {
        SCOPED_TRACE(deviceTypeToString(backend));
        auto config = request(source, backend);
        config.automatic_planning.only_strategies =
            std::vector{OrchestrationStrategy::SingleDevice, OrchestrationStrategy::TensorParallel};
        auto observed = inventory(backend);
        for (int order = 0; order != 2; ++order)
        {
            size_t multi_device = 0;
            const auto result = AutomaticOrchestrationPlanner::select(config, source, observed, memory(), {64, 128},
                [&](const auto &candidate, const auto &work) {
                    // Rank count alone cannot distinguish these alternatives.
                    EXPECT_EQ(candidate.rankPlans().size(), 1u);
                    if (candidate.strategy() == OrchestrationStrategy::TensorParallel)
                    {
                        ++multi_device;
                        EXPECT_EQ(candidate.devicePlans().size(), 2u);
                    }
                    return OrchestrationCostEstimate(work, 1, .01, "synthetic equal request costs");
                });
            EXPECT_EQ(multi_device, 1u);
            EXPECT_EQ(result.candidate().strategy(), OrchestrationStrategy::SingleDevice);
            std::reverse(observed.ranks.front().gpus.begin(), observed.ranks.front().gpus.end());
            observed.buildNodeAggregations();
        }
    }
}

TEST(AutomaticOrchestrationPlanner, MultiDeviceMustEarnCommunicationCostAndStillRespectsConstraints)
{
    test::PlanningGGUFFixture file;
    PlanningModelSource source(file.path());
    for (const auto backend : {DeviceType::CPU, DeviceType::CUDA, DeviceType::ROCm})
    {
        SCOPED_TRACE(deviceTypeToString(backend));
        auto config = request(source, backend);
        config.automatic_planning.only_strategies =
            std::vector{OrchestrationStrategy::SingleDevice, OrchestrationStrategy::TensorParallel};
        const auto observed = inventory(backend);
        const auto cost = [](double multi_link_seconds) {
            return [multi_link_seconds](const auto &candidate, const auto &work) {
                const bool single = candidate.strategy() == OrchestrationStrategy::SingleDevice;
                // Synthetic costs isolate selection policy: a link-free single
                // device loses only when the parallel compute saving pays for
                // its communication. These are not measurements of hardware.
                const double compute = single ? .02 : .01;
                const double communication = single ? 0 : multi_link_seconds;
                return OrchestrationCostEstimate(work, (compute + communication) * work.promptTokens(),
                    compute + communication, "synthetic compute plus explicit interconnect cost");
            };
        };
        const auto expensive_link = AutomaticOrchestrationPlanner::select(
            config, source, observed, memory(), {64, 128}, cost(.02));
        EXPECT_EQ(expensive_link.candidate().strategy(), OrchestrationStrategy::SingleDevice);
        const auto profitable_tp = AutomaticOrchestrationPlanner::select(
            config, source, observed, memory(), {64, 128}, cost(.001));
        EXPECT_EQ(profitable_tp.candidate().strategy(), OrchestrationStrategy::TensorParallel);
        // A default preference never broadens an explicit strategy constraint.
        config.automatic_planning.only_strategies = std::vector{OrchestrationStrategy::TensorParallel};
        const auto required_tp = AutomaticOrchestrationPlanner::select(
            config, source, observed, memory(), {64, 128}, cost(.02));
        EXPECT_EQ(required_tp.candidate().strategy(), OrchestrationStrategy::TensorParallel);
    }
}

TEST(AutomaticOrchestrationPlanner, EqualCostsPreferNarrowerTPWithoutDisablingWiderCandidates)
{
    test::PlanningGGUFFixture file;
    PlanningModelSource source(file.path());
    for (const auto backend : {DeviceType::CUDA, DeviceType::ROCm})
    {
        auto config = request(source, backend);
        config.automatic_planning.only_strategies = std::vector{OrchestrationStrategy::TensorParallel};
        auto observed = inventory(backend);
        auto third = observed.ranks.front().gpus.back();
        third.local_device_id = 1;
        third.uuid = "third";
        observed.ranks.front().gpus.push_back(third);
        observed.buildNodeAggregations();
        bool saw_three = false;
        const auto result = AutomaticOrchestrationPlanner::select(config, source, observed, memory(), {64, 128},
            [&](const auto &candidate, const auto &work) {
                saw_three |= candidate.devicePlans().size() == 3;
                return OrchestrationCostEstimate(work, 1, .01, "synthetic tie across TP widths");
            });
        EXPECT_TRUE(saw_three);
        EXPECT_EQ(result.candidate().devicePlans().size(), 2u);
    }
}

TEST(AutomaticOrchestrationPlanner, AutomaticCandidatesExcludeGDNIncompatibleTensorParallelWidths)
{
    test::PlanningGGUFFixture file;
    PlanningModelSource source(file.path());
    auto profile = source.metadata().memoryProfile();
    // Eight Q heads split over three ranks are [4, 2, 2] under the production
    // GQA-aware splitter.  GDN's two key heads therefore cross the second
    // boundary; degree two remains dependency-closed and must remain searchable.
    profile.gdn_group_count = 2;
    profile.gdn_time_step_rank = 8;
    PlanningModelMetadata hybrid(std::move(profile), source.metadata().mainLayerCount());

    for (const auto backend : {DeviceType::CUDA, DeviceType::ROCm})
    {
        auto config = request(source, backend);
        config.automatic_planning.only_strategies = std::vector{OrchestrationStrategy::TensorParallel};
        auto observed = inventory(backend);
        auto third = observed.ranks.front().gpus.back();
        third.local_device_id = 1;
        third.uuid = "third";
        observed.ranks.front().gpus.push_back(third);
        observed.buildNodeAggregations();

        std::set<size_t> widths;
        visitAutomaticOrchestrationCandidates(config, hybrid, observed,
            [&](AutomaticOrchestrationCandidate candidate) {
                EXPECT_EQ(candidate.strategy, OrchestrationStrategy::TensorParallel);
                widths.insert(candidate.config.tp_devices.size());
            });
        EXPECT_EQ(widths, (std::set<size_t>{2u}));
    }
}

TEST(AutomaticOrchestrationPlanner, StartupRejectsImpossibleWorkloadBeforeOpeningModel)
{
    OrchestrationConfig config;
    config.model_path = "/__must_not_open_invalid_workload__/model.gguf";
    config.max_seq_len = 512;
    config.automatic_planning.workload = OrchestrationPlanningWorkload(512, 1);
    try
    {
        (void)AutomaticPlanningStartup::run(config, inventory(DeviceType::CUDA), {});
        FAIL() << "Impossible workload was admitted";
    }
    catch (const std::runtime_error &error)
    {
        EXPECT_NE(std::string(error.what()).find("workload exceeds"), std::string::npos);
    }
}

TEST(AutomaticOrchestrationPlanner, StartupRejectsMissingEvidenceAndForeignDiscoveryBeforePricing)
{
    test::PlanningGGUFFixture file;
    PlanningModelSource source(file.path());
    const auto config = request(source, DeviceType::CUDA);
    const auto observed = inventory(DeviceType::CUDA);
    size_t prices = 0;
    const auto empty = [](const AutomaticPlanningPreparation &)
        -> std::optional<AutomaticOrchestrationPlanner::Evaluate> { return std::nullopt; };
    const auto invalid = [](const AutomaticPlanningPreparation &)
        -> std::optional<AutomaticOrchestrationPlanner::Evaluate> { return AutomaticOrchestrationPlanner::Evaluate{}; };
    const auto failed = [](const AutomaticPlanningPreparation &)
        -> std::optional<AutomaticOrchestrationPlanner::Evaluate> { throw std::runtime_error("Cost preparation failed"); };
    for (const AutomaticPlanningStartup::Prepare &prepare : {AutomaticPlanningStartup::Prepare(empty),
            AutomaticPlanningStartup::Prepare(invalid), AutomaticPlanningStartup::Prepare(failed)})
        EXPECT_THROW(AutomaticPlanningStartup::run(config, observed, {}, prepare), std::runtime_error);
    EXPECT_THROW(AutomaticPlanningStartup::run(config, inventory(DeviceType::CPU), {},
        [&](const AutomaticPlanningPreparation &) -> std::optional<AutomaticOrchestrationPlanner::Evaluate> {
            ++prices;
            return [](const auto &, const auto &workload) {
                ADD_FAILURE() << "Foreign discovery must fail before pricing";
                return OrchestrationCostEstimate(workload, 1, .01, "unreachable unit oracle");
            };
        }), std::runtime_error);
    EXPECT_EQ(prices, 0u);
}

TEST(AutomaticOrchestrationPlanner, PhysicalRejectionsAreObservedButNeverPriced)
{
    test::PlanningGGUFFixture file;
    PlanningModelSource source(file.path());
    auto observed = inventory(DeviceType::CUDA);
    observed.ranks.front().gpus.front().free_memory_bytes = 0;
    observed.buildNodeAggregations();
    size_t rejected = 0, priced = 0;
    const auto selected = AutomaticOrchestrationPlanner::select(request(source, DeviceType::CUDA), source,
        observed, memory(), {64, 128}, [&](const auto &candidate, const auto &work) {
            ++priced;
            EXPECT_TRUE(smaller(candidate));
            return OrchestrationCostEstimate(work, 1.0, 0.01, "fixture");
        }, [&](auto strategy, auto detail) {
            ++rejected;
            EXPECT_EQ(strategy, OrchestrationStrategy::SingleDevice);
            EXPECT_FALSE(detail.empty());
        });
    EXPECT_EQ(rejected, 1u);
    EXPECT_EQ(priced, 1u);
    EXPECT_EQ(selected.counts().capacity_rejected, rejected);
    EXPECT_EQ(selected.counts().evaluated, priced);
}

TEST(AutomaticOrchestrationPlanner, ExhaustionDoesNotInventAnotherBackend)
{
    test::PlanningGGUFFixture file;
    PlanningModelSource source(file.path());
    auto observed = inventory(DeviceType::CUDA);
    for (auto &gpu : observed.ranks.front().gpus) gpu.free_memory_bytes = 0;
    observed.buildNodeAggregations();
    const auto cost = [](const auto &, const auto &work) {
        ADD_FAILURE() << "An exhausted or forbidden CPU candidate reached the evaluator";
        return OrchestrationCostEstimate(work, 1.0, 0.01, "fixture");
    };
    EXPECT_THROW(AutomaticOrchestrationPlanner::select(request(source, DeviceType::CUDA), source,
        observed, memory(), {64, 128}, cost), PhysicalMemoryCapacityExhausted);
    observed.ranks.front().gpus.clear();
    observed.buildNodeAggregations();
    EXPECT_THROW(AutomaticOrchestrationPlanner::select(request(source, DeviceType::CUDA), source,
        observed, memory(), {64, 128}, cost), std::invalid_argument);
}

TEST(AutomaticOrchestrationPlanner, EvaluatorFailuresAndWrongWorkloadsCannotSelectAnotherCandidate)
{
    test::PlanningGGUFFixture file;
    PlanningModelSource source(file.path());
    const auto config = request(source, DeviceType::CPU);
    const auto observed = inventory(DeviceType::CPU);
    EXPECT_THROW(AutomaticOrchestrationPlanner::select(config, source, observed, memory(), {64, 128}, {}), std::invalid_argument);
    int calls = 0;
    const auto broken = [&](const auto &, const auto &) -> OrchestrationCostEstimate {
        ++calls;
        throw PhysicalMemoryCapacityExhausted("A cost failure is not an admission rejection");
    };
    EXPECT_THROW(AutomaticOrchestrationPlanner::select(config, source, observed, memory(), {64, 128}, broken), PhysicalMemoryCapacityExhausted);
    EXPECT_EQ(calls, 1);
    const auto stale = [](const auto &, const auto &) { return OrchestrationCostEstimate({65, 128}, 1.0, 0.01, "wrong workload"); };
    EXPECT_THROW(AutomaticOrchestrationPlanner::select(config, source, observed, memory(), {64, 128}, stale), std::invalid_argument);
    EXPECT_THROW(AutomaticOrchestrationPlanner::select(config, source, observed, memory(), {1024, 1}, stale), std::invalid_argument);
    EXPECT_THROW(AutomaticOrchestrationPlanner::select(config, source, observed, memory(),
        {std::numeric_limits<int>::max(), std::numeric_limits<int>::max()}, stale), std::invalid_argument);
}

TEST(AutomaticOrchestrationPlanner, HintsBreakTiesButCannotOverrideLowerCost)
{
    test::PlanningGGUFFixture file;
    PlanningModelSource source(file.path());
    auto observed = inventory(DeviceType::CUDA);
    observed.ranks.front().gpus.back().type = DeviceType::ROCm;
    observed.buildNodeAggregations();
    auto config = request(source, DeviceType::CUDA);
    config.automatic_planning.only_backends = std::vector{DeviceType::CUDA, DeviceType::ROCm};
    config.automatic_planning.prefer_backend = DeviceType::ROCm;
    const auto tied = [](const auto &, const auto &work) { return OrchestrationCostEstimate(work, 1.0, 0.01, "fixture tie"); };
    const auto hinted = AutomaticOrchestrationPlanner::select(config, source, observed, memory(), {64, 128}, tied);
    EXPECT_TRUE(hinted.candidate().config().device_for_this_rank->isROCm());
    const auto untied = [](const auto &candidate, const auto &work) {
        return OrchestrationCostEstimate(work, candidate.config().device_for_this_rank->isCUDA() ? 0.9 : 1.0, 0.01, "fixture costs");
    };
    const auto faster = AutomaticOrchestrationPlanner::select(config, source, observed, memory(), {64, 128}, untied);
    EXPECT_TRUE(faster.candidate().config().device_for_this_rank->isCUDA());

    config.automatic_planning.only_backends = std::vector{DeviceType::CUDA};
    EXPECT_THROW(AutomaticOrchestrationPlanner::select(config, source, observed, memory(), {64, 128}, tied), std::invalid_argument);
}

TEST(AutomaticOrchestrationPlanner, ExactTiesHaveAStableOrderAndStrategyHintsArePreserved)
{
    test::PlanningGGUFFixture file;
    PlanningModelSource source(file.path());
    auto config = request(source, DeviceType::CUDA);
    auto observed = inventory(DeviceType::CUDA);
    const auto cost = [](const auto &, const auto &work) { return OrchestrationCostEstimate(work, 1.0, 0.01, "fixture tie"); };
    const auto first = AutomaticOrchestrationPlanner::select(config, source, observed, memory(), {64, 128}, cost);
    std::reverse(observed.ranks.front().gpus.begin(), observed.ranks.front().gpus.end());
    observed.buildNodeAggregations();
    const auto reversed = AutomaticOrchestrationPlanner::select(config, source, observed, memory(), {64, 128}, cost);
    EXPECT_EQ(serializeOrchestrationConfig(first.candidate().config()), serializeOrchestrationConfig(reversed.candidate().config()));
    config.automatic_planning.only_strategies =
        std::vector{OrchestrationStrategy::SingleDevice, OrchestrationStrategy::TensorParallel};
    config.automatic_planning.prefer_strategy = OrchestrationStrategy::TensorParallel;
    const auto hinted = AutomaticOrchestrationPlanner::select(config, source, observed, memory(), {64, 128}, cost);
    EXPECT_EQ(hinted.candidate().strategy(), OrchestrationStrategy::TensorParallel);
}

TEST(AutomaticOrchestrationPlanner, CompilerAndDiagnosticFailuresRemainFatal)
{
    test::PlanningGGUFFixture file;
    PlanningModelSource source(file.path());
    auto config = request(source, DeviceType::CUDA);
    const auto cost = [](const auto &, const auto &work) { return OrchestrationCostEstimate(work, 1.0, 0.01, "fixture"); };
    auto invalid_memory = memory();
    invalid_memory.prefill_bucket_rows.clear();
    EXPECT_THROW(AutomaticOrchestrationPlanner::select(config, source, inventory(DeviceType::CUDA), invalid_memory, {64, 128}, cost), std::invalid_argument);
    auto invalid_config = config;
    invalid_config.activation_precision = "fp16";
    EXPECT_THROW(AutomaticOrchestrationPlanner::select(invalid_config, source, inventory(DeviceType::CUDA), memory(), {64, 128}, cost), std::invalid_argument);
    auto observed = inventory(DeviceType::CUDA);
    observed.ranks.front().gpus.front().free_memory_bytes = 0;
    observed.buildNodeAggregations();
    EXPECT_THROW(AutomaticOrchestrationPlanner::select(config, source, observed, memory(), {64, 128}, cost,
        [](auto, auto) { throw std::logic_error("observer failed"); }), std::logic_error);
    config.device_for_this_rank = GlobalDeviceAddress::cuda(2);
    EXPECT_THROW(AutomaticOrchestrationPlanner::select(config, source, observed, memory(), {64, 128}, cost), std::invalid_argument);
}

TEST(AutomaticOrchestrationPlanner, RemoteOverlaySelectionPreservesDynamicMTPAndPrecisionPolicies)
{
    for (const auto format : {GGUFTensorType::F32, GGUFTensorType::F16, GGUFTensorType::BF16})
    for (const auto backend : {DeviceType::CUDA, DeviceType::ROCm})
    for (const int remote_hosts : {1, 2})
    {
        SCOPED_TRACE(::testing::Message() << "format=" << static_cast<int>(format)
            << " backend=" << deviceTypeToString(backend) << " remotes=" << remote_hosts);
        test::PlanningGGUFFixture file(true, true, format);
        PlanningModelSource source(file.path());
        auto observed = inventory(backend);
        observed.ranks.front().gpus.resize(1);
        for (int rank = 1; rank <= remote_hosts; ++rank)
        {
            auto cpu = observed.ranks.front();
            cpu.rank = cpu.node_id = rank;
            cpu.hostname = "remote-" + std::to_string(rank);
            cpu.gpus.clear();
            observed.ranks.push_back(std::move(cpu));
        }
        observed.world_size = static_cast<int>(observed.ranks.size());
        observed.buildNodeAggregations();
        auto config = request(source, backend);
        config.automatic_planning.only_backends = std::vector{backend, DeviceType::CPU};
        config.automatic_planning.only_strategies = std::vector{OrchestrationStrategy::ExpertOverlay};
        config.moe_rebalance.mode = MoERebalanceRuntimeMode::Dynamic;
        config.mtp.enabled = true;
        config.mtp.graph_capacity_draft_tokens = 15;
        config.mtp.depth_policy.mode = MTPDepthPolicyMode::Dynamic;
        config.mtp.depth_policy.max_depth = 15;
        // This tiny descriptor has head_dim=32, outside TurboQuant's supported
        // geometry. Use a non-default valid KV policy; do not waive admission.
        config.kv_cache_precision = "fp32";
        config.prefill_max_bucket_size = 64;
        config.prefix_cache.ram_budget_bytes = 64ull << 20;
        config.prefix_cache.disk_budget_bytes = 128ull << 20;
        config.prefix_cache.disk_dir = "saved prefix tier";
        const auto selected = AutomaticOrchestrationPlanner::select(config, source, observed, memory(), {64, 128},
            [&](const auto &candidate, const auto &work) {
                EXPECT_TRUE(candidate.overlayCapacity());
                // This fixture ranks by declared costs only. Host count is
                // deliberately not a production speed heuristic.
                return OrchestrationCostEstimate(work,
                    candidate.membership().selection().size() == remote_hosts + 1 ? 0.1 : 0.2,
                    0.01, "synthetic remote decision oracle");
            });
        EXPECT_EQ(selected.candidate().membership().selection().size(), remote_hosts + 1);
        const auto applied = deserializeOrchestrationConfig(serializeOrchestrationConfig(selected.candidate().config()));
        EXPECT_EQ(applied.moe_rebalance.mode, MoERebalanceRuntimeMode::Dynamic);
        EXPECT_TRUE(applied.mtp.enabled);
        EXPECT_EQ(applied.mtp.graph_capacity_draft_tokens, 15);
        EXPECT_EQ(applied.mtp.depth_policy.mode, MTPDepthPolicyMode::Dynamic);
        EXPECT_EQ(applied.mtp.depth_policy.max_depth, 15);
        EXPECT_EQ(applied.kv_cache_precision, "fp32");
        EXPECT_EQ(applied.prefill_max_bucket_size, 64);
        EXPECT_TRUE(applied.prefix_cache.enabled);
        EXPECT_EQ(applied.prefix_cache.storage_mode, PrefixCacheStorageMode::Tiered);
        EXPECT_EQ(applied.prefix_cache.ram_budget_bytes, 64ull << 20);
        EXPECT_EQ(applied.prefix_cache.disk_budget_bytes, 128ull << 20);
        EXPECT_EQ(applied.prefix_cache.disk_dir, "saved prefix tier");
        ASSERT_TRUE(applied.moe_routed_expert_plan);
        EXPECT_TRUE(applied.moe_routed_expert_plan->usesExpertOverlayAuthority());
    }
}
