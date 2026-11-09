/**
 * @file Test__AutomaticOrchestrationPlanner.cpp
 * @brief Device-free proof that complete admission precedes cost-ranked selection.
 *
 * Costs below are deliberately synthetic decision oracles, not speed claims.
 * The model directory, candidate compiler, memory authority and apply codec
 * are real production components. No tensor payload or accelerator is used.
 */
#include "../../utils/PlanningHostServiceFixture.h"
#include "planning/AutomaticOrchestrationPlanner.h"
#include "planning/ObservedPeerGrouping.h"
#include "planning/AutomaticOrchestrationCandidates.h"
#include "planning/AutomaticPlanningStartup.h"
#include "planning/PhysicalMemoryCapacityExhausted.h"
#include "config/OrchestrationConfigDocument.h"
#include "../../utils/PlanningGGUFFixture.h"
#include <gtest/gtest.h>
#include <algorithm>
#include <array>
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
            .maximum_cached_prefill_buckets = 8, .retained_host_services = test::planningHostServiceBudgets(2)};
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

/** @brief Plan/default-auto serve reject missing predictors before hardware measurement. */
TEST(AutomaticOrchestrationPlanner, MissingMTPWeightsFailBeforeEvidencePreparation)
{
    for (const bool moe : {false, true})
    {
        test::PlanningGGUFFixture file(moe, false);
        PlanningModelSource source(file.path());
        for (const auto backend : {DeviceType::CPU, DeviceType::CUDA, DeviceType::ROCm})
        for (const bool active : {false, true})
        {
            SCOPED_TRACE(::testing::Message() << "moe=" << moe << " backend="
                << deviceTypeToString(backend) << " active=" << active);
            auto config = request(source, backend);
            config.mtp.enabled = active;
            config.mtp.graph_capacity_draft_tokens = 15;
            const auto original = serializeOrchestrationConfig(config);
            // This test has no MPI context. Keep discovery process-local so a
            // communicator mismatch cannot mask learned-weight admission.
            auto observed = inventory(backend);
            observed.world_size = 1;
            observed.ranks.resize(1);
            observed.buildNodeAggregations();
            size_t preparations = 0;
            try
            {
                (void)AutomaticPlanningStartup::run(config, observed, {},
                    [&](const AutomaticPlanningPreparation &)
                        -> std::optional<AutomaticOrchestrationPlanner::Evaluate> {
                        ++preparations;
                        throw std::logic_error("Missing MTP reached hardware preparation");
                    });
                FAIL() << "A plain GGUF reached automatic selection with retained MTP";
            }
            catch (const std::runtime_error &error)
            {
                EXPECT_NE(std::string(error.what()).find("no MTP/nextn metadata or tensors"), std::string::npos)
                    << error.what();
            }
            EXPECT_EQ(preparations, 0u);
            EXPECT_EQ(serializeOrchestrationConfig(config), original);
        }
    }
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

// Requires a compiled CUDA/ROCm backend family: its subject prices or
// constructs GPU-native objects that do not exist in a CPU-only build.
#if defined(HAVE_CUDA) || defined(HAVE_ROCM)
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
#endif // defined(HAVE_CUDA) || defined(HAVE_ROCM)

// Requires a compiled CUDA/ROCm backend family: its subject prices or
// constructs GPU-native objects that do not exist in a CPU-only build.
#if defined(HAVE_CUDA) || defined(HAVE_ROCM)
/**
 * @brief Full auto selection admits three physical tiers before ranking them.
 *
 * Candidate enumeration alone cannot prove that model-aware compilation,
 * PhysicalMemoryAuthority and the final apply document accept a deeper plan.
 * The synthetic cost oracle selects the CUDA-continuation ordering; it makes
 * no claim about the relative speed of these fixture devices.
 */
TEST(AutomaticOrchestrationPlanner, MultiTierOverlayAdmissionAndSelection)
{
    test::PlanningGGUFFixture file(true);
    PlanningModelSource source(file.path());
    auto observed = inventory(DeviceType::CUDA);
    observed.ranks.front().node_id = 0;
    observed.ranks.front().hostname = "shared-host";
    auto rocm_rank = observed.ranks.front();
    rocm_rank.rank = 1;
    rocm_rank.local_rank = 1;
    rocm_rank.cpu.numa_node = 4;
    for (auto &gpu : rocm_rank.gpus)
    {
        gpu.type = DeviceType::ROCm;
        gpu.uuid = "rocm-" + gpu.uuid;
        gpu.numa_node = 4;
    }
    // These fixture domains have observed no direct peer edges. Supply the
    // complete directed matrices: absent discovery is not negative evidence.
    auto &cuda_rank = observed.ranks.front();
    cuda_rank.p2p_cuda_count = static_cast<int>(cuda_rank.gpus.size());
    cuda_rank.p2p_cuda.assign(cuda_rank.gpus.size() * cuda_rank.gpus.size(), false);
    rocm_rank.p2p_rocm_count = static_cast<int>(rocm_rank.gpus.size());
    rocm_rank.p2p_rocm.assign(rocm_rank.gpus.size() * rocm_rank.gpus.size(), false);
    observed.ranks.push_back(std::move(rocm_rank));
    observed.world_size = 2;
    observed.buildNodeAggregations();

    auto config = request(source, DeviceType::CUDA);
    config.automatic_planning.only_backends =
        std::vector{DeviceType::CPU, DeviceType::CUDA, DeviceType::ROCm};
    config.automatic_planning.only_strategies =
        std::vector{OrchestrationStrategy::ExpertOverlay};
    config.automatic_planning.device_counts = {
        {DeviceType::CPU, 2}, {DeviceType::CUDA, 2}, {DeviceType::ROCm, 2}};
    // The fixture's small head geometry is outside TurboQuant admission.
    config.kv_cache_precision = "fp32";
    config.prefill_max_bucket_size = 64;
    bool priced_desired = false;
    const auto selected = AutomaticOrchestrationPlanner::select(
        config, source, observed, memory(), {64, 128},
        [&](const AdmittedOrchestrationCandidate &candidate,
            const OrchestrationPlanningWorkload &work)
        {
            EXPECT_TRUE(candidate.physicalAdmission().plan().fits());
            EXPECT_TRUE(candidate.overlayCapacity());
            const auto &domains = candidate.config().moe_routed_expert_plan->domains;
            const bool desired = domains.size() == 3u &&
                domains[0].participants.front().isCUDA() &&
                domains[1].participants.front().isROCm() &&
                domains[2].participants.front().isCPU();
            priced_desired |= desired;
            return OrchestrationCostEstimate(work, desired ? 0.1 : 1.0,
                0.01, "synthetic three-tier decision oracle");
        });
    EXPECT_TRUE(priced_desired);
    EXPECT_GE(selected.counts().evaluated, 1u);
    const auto &plan = *selected.candidate().config().moe_routed_expert_plan;
    ASSERT_EQ(plan.domains.size(), 3u);
    EXPECT_TRUE(plan.domains[0].participants.front().isCUDA());
    EXPECT_TRUE(plan.domains[1].participants.front().isROCm());
    EXPECT_TRUE(plan.domains[2].participants.front().isCPU());
    const auto saved = serializeOrchestrationConfig(selected.candidate().config());
    EXPECT_EQ(serializeOrchestrationConfig(deserializeOrchestrationConfig(saved)), saved);
}
#endif // defined(HAVE_CUDA) || defined(HAVE_ROCM)

namespace {
/** @return Original sparse, interleaved endpoints with two observed local PCI islands. */
RankInventory peerIslands(std::size_t count=8)
{
    RankInventory rank;
    rank.rank=0;rank.node_id=0;rank.hostname="peer-host";rank.cpu.numa_node=3;
    rank.cpu.memory_bytes=rank.cpu.free_memory_bytes=128ull<<30;
    rank.cpu_cores=rank.cpu.compute_units=8;rank.cpu_worker_threads=2;rank.numa_nodes=1;
    const std::array ordinals{91,7,53,2,87,11,61,3,103,17,121,19};
    for(std::size_t i=0;i<count;++i) {
        DeviceInfo gpu;
        gpu.type=DeviceType::ROCm;gpu.local_device_id=ordinals.at(i);gpu.uuid="card-"+std::to_string(i);
        gpu.numa_node=3;gpu.memory_bytes=64ull<<30;gpu.free_memory_bytes=48ull<<30;gpu.compute_units=64;
        gpu.native_isa="gfx906:sramecc+:xnack-";
        gpu.pcie_upstream_bdfs={i%2?"0000:af:00.0":"0000:86:00.0","0000:80:00.0"};
        rank.gpus.push_back(std::move(gpu));
    }
    rank.p2p_rocm_count=static_cast<int>(count);rank.p2p_rocm.assign(count*count,true);
    return rank;
}
/** @return Original group endpoint sets independent of semantic stage order. */
std::set<std::set<int>> groupOrdinals(const std::vector<std::vector<DeviceId>>& groups)
{
    std::set<std::set<int>> result;
    for(const auto& group:groups) {std::set<int> ids;for(auto id:group) ids.insert(id.ordinal);result.insert(std::move(ids));}
    return result;
}
}

TEST(ObservedPeerGrouping, AllAccessibleSparseInterleavedIslandsKeepOriginalFourPlusFour)
{
    auto rank=peerIslands();
    std::vector<DeviceId> eligible;
    for(const auto& gpu:rank.gpus) eligible.push_back({gpu.type,gpu.local_device_id});
    const std::array<std::size_t,2> sizes{4,4};
    const auto groups=ObservedPeerGrouping(rank).select(eligible,sizes);
    EXPECT_EQ(groupOrdinals(groups),(std::set<std::set<int>>{{91,53,87,61},{7,2,11,3}}));
    // Shuffle both discovery order and matrix positions, preserving the actual
    // complete graph. UUID authority, not ordinal/discovery adjacency, wins ties.
    std::reverse(rank.gpus.begin(),rank.gpus.end());
    std::reverse(eligible.begin(),eligible.end());
    EXPECT_EQ(ObservedPeerGrouping(rank).select(eligible,sizes),groups);
}

TEST(ObservedPeerGrouping, MutualFactsNeverInventAsymmetricPartialMissingOrCrossDomainEdges)
{
    auto rank=peerIslands(4);
    const DeviceId a{DeviceType::ROCm,91},b{DeviceType::ROCm,7},c{DeviceType::ROCm,53},d{DeviceType::ROCm,2};
    rank.p2p_rocm.assign(16,false);
    rank.p2p_rocm[0*4+1]=true; // One directed edge never establishes mutual access.
    rank.p2p_rocm[0*4+2]=rank.p2p_rocm[2*4+0]=true;
    const ObservedPeerGrouping facts(rank);
    EXPECT_FALSE(facts.mutuallyAccessible(a,b));EXPECT_TRUE(facts.mutuallyAccessible(a,c));
    EXPECT_EQ(facts.quality(std::array{a,b,c}).fully_connected_pairs,0u);
    EXPECT_EQ(facts.quality(std::array{a,b,c}).mutual_pairs,1u);
    EXPECT_EQ(groupOrdinals(facts.select(std::array{a,b,c,d},std::array<std::size_t,2>{2,2})),
        (std::set<std::set<int>>{{91,53},{7,2}}));
    auto shuffled=rank;
    std::reverse(shuffled.gpus.begin(),shuffled.gpus.end());
    for(std::size_t i=0;i<4;++i) for(std::size_t j=0;j<4;++j)
        shuffled.p2p_rocm[i*4+j]=rank.p2p_rocm[(3-i)*4+(3-j)];
    EXPECT_EQ(groupOrdinals(ObservedPeerGrouping(shuffled).select(std::array{d,c,b,a},std::array<std::size_t,2>{2,2})),
        (std::set<std::set<int>>{{91,53},{7,2}}));
    rank.p2p_rocm.clear();
    EXPECT_EQ(ObservedPeerGrouping(rank).quality(std::array{a,c}),ObservedPeerGroupQuality{});
    // Missing capability still permits a real homogeneous native CCL group.
    EXPECT_EQ(ObservedPeerGrouping(rank).select(std::array{a,b,c,d},std::array<std::size_t,1>{4}).front().size(),4u);
    rank.p2p_rocm.assign(15,true); // Structurally incomplete observation is unknown.
    EXPECT_FALSE(ObservedPeerGrouping(rank).mutuallyAccessible(a,c));
    rank.p2p_rocm.assign(16,true);
    rank.gpus[2].numa_node=7;
    EXPECT_FALSE(ObservedPeerGrouping(rank).mutuallyAccessible(a,c));
    EXPECT_THROW(ObservedPeerGrouping(rank).select(std::array{a,c},std::array<std::size_t,1>{2}),std::invalid_argument);
    rank.gpus[2].numa_node=3;rank.gpus[2].type=DeviceType::CUDA;
    const DeviceId cuda{DeviceType::CUDA,53};
    EXPECT_FALSE(ObservedPeerGrouping(rank).mutuallyAccessible(a,cuda));
    EXPECT_THROW(ObservedPeerGrouping(rank).select(std::array{a,cuda},std::array<std::size_t,1>{2}),std::invalid_argument);
}

TEST(ObservedPeerGrouping, RequestedSizesAndDeterministicTiesHaveNoMachineCardinalityConstants)
{
    for(std::size_t count=1;count<=11;++count) {
        auto rank=peerIslands(count);
        std::vector<DeviceId> eligible;for(const auto& gpu:rank.gpus) eligible.push_back({gpu.type,gpu.local_device_id});
        std::vector<std::size_t> sizes(count/3,3);if(count%3) sizes.push_back(count%3);
        const auto selected=ObservedPeerGrouping(rank).select(eligible,sizes);
        ASSERT_EQ(selected.size(),sizes.size());
        std::set<DeviceId> distinct;
        for(std::size_t i=0;i<sizes.size();++i) {EXPECT_EQ(selected[i].size(),sizes[i]);distinct.insert(selected[i].begin(),selected[i].end());}
        EXPECT_EQ(distinct.size(),count);
        // Remove locality and capability facts: legal native groups retain
        // deterministic UUID tie order without fabricated P2P eligibility.
        rank.p2p_rocm.clear();for(auto& gpu:rank.gpus) gpu.pcie_upstream_bdfs.clear();
        const auto unobserved=ObservedPeerGrouping(rank).select(eligible,sizes);
        std::reverse(eligible.begin(),eligible.end());
        EXPECT_EQ(ObservedPeerGrouping(rank).select(eligible,sizes),unobserved);
    }
    const auto rank=peerIslands(2);const ObservedPeerGrouping facts(rank);
    EXPECT_THROW(facts.select(std::array{DeviceId::rocm(91)},std::array<std::size_t,1>{0}),std::invalid_argument);
    EXPECT_THROW(facts.select(std::array{DeviceId::rocm(91)},std::array<std::size_t,1>{2}),std::invalid_argument);
    EXPECT_THROW(facts.select(std::array{DeviceId::rocm(91),DeviceId::rocm(91)},std::array<std::size_t,1>{1}),std::invalid_argument);
}

TEST(AutomaticOrchestrationPlanner, ObservedPeerIslandsChooseActualOriginalDenseFourPlusFourAfterAdmission)
{
    test::PlanningGGUFFixture file;
    PlanningModelSource source(file.path());
    ClusterInventory observed;observed.world_size=1;observed.ranks.push_back(peerIslands());observed.buildNodeAggregations();
    auto config=request(source,DeviceType::ROCm);
    config.automatic_planning.only_strategies=std::vector{OrchestrationStrategy::PipelineParallel};
    config.automatic_planning.device_counts={{DeviceType::ROCm,8}};
    config.kv_cache_precision="fp32";config.prefill_max_bucket_size=64;
    std::size_t candidates=0,nonlocal=0;
    const auto selected=AutomaticOrchestrationPlanner::select(config,source,observed,memory(),{64,128},
        [&](const AdmittedOrchestrationCandidate& candidate,const auto& work) {
            ++candidates;
            const auto& domains=candidate.config().domain_definitions;
            const bool balanced=domains.size()==2 && domains[0].devices.size()==4 && domains[1].devices.size()==4;
            if(balanced) {
                std::vector<DeviceId> ids;for(const auto& address:domains[0].devices) ids.push_back({address.device_type,address.device_ordinal});
                nonlocal+=ObservedPeerGrouping(observed.ranks[0]).quality(ids).negative_hops < -12;
            }
            return OrchestrationCostEstimate(work,balanced?1:2,.01,"synthetic equal containing-native-group proxy across partitions");
        });
    EXPECT_GT(candidates,1u);EXPECT_GT(nonlocal,0u);
    std::vector<std::vector<DeviceId>> actual;
    for(const auto& domain:selected.candidate().config().domain_definitions) {
        std::vector<DeviceId> ids;for(const auto& address:domain.devices) ids.push_back({address.device_type,address.device_ordinal});
        actual.push_back(std::move(ids));
    }
    EXPECT_EQ(groupOrdinals(actual),(std::set<std::set<int>>{{91,53,87,61},{7,2,11,3}}));
}

TEST(AutomaticOrchestrationPlanner, ObservedPeerPreferencePreservesBetterCostAndNativeNoPeerCandidates)
{
    test::PlanningGGUFFixture file;PlanningModelSource source(file.path());
    for(bool available:{false,true}) {
        auto rank=peerIslands(3);
        if(!available) rank.p2p_rocm.clear();
        ClusterInventory observed;observed.world_size=1;observed.ranks.push_back(rank);observed.buildNodeAggregations();
        auto config=request(source,DeviceType::ROCm);
        config.automatic_planning.only_strategies=std::vector{OrchestrationStrategy::TensorParallel};
        config.automatic_planning.device_counts={{DeviceType::ROCm,2}};
        config.kv_cache_precision="fp32";
        std::size_t priced=0;
        const auto selected=AutomaticOrchestrationPlanner::select(config,source,observed,memory(),{64,128},
            [&](const AdmittedOrchestrationCandidate& candidate,const auto& work) {
                ++priced;std::set<int> ids;for(const auto& p:candidate.devicePlans()) ids.insert(p.device.ordinal);
                return OrchestrationCostEstimate(work,ids==std::set<int>{91,7}?.5:1,.01,"synthetic strictly better qualified cost dominates physical preference");
            });
        EXPECT_EQ(priced,3u);
        std::set<int> ids;for(const auto& p:selected.candidate().devicePlans()) ids.insert(p.device.ordinal);
        EXPECT_EQ(ids,(std::set<int>{91,7}));
    }
}

TEST(AutomaticOrchestrationPlanner, ObservedPeerPreferenceCannotBypassPhysicalCapacityAdmission)
{
    test::PlanningGGUFFixture file;PlanningModelSource source(file.path());
    auto rank=peerIslands(3);
    rank.p2p_rocm.assign(9,false);rank.p2p_rocm[2]=rank.p2p_rocm[6]=true;
    rank.gpus[2].free_memory_bytes=1; // The only eligible peer edge cannot fit its source tensors.
    ClusterInventory observed;observed.world_size=1;observed.ranks.push_back(rank);observed.buildNodeAggregations();
    auto config=request(source,DeviceType::ROCm);
    config.automatic_planning.only_strategies=std::vector{OrchestrationStrategy::TensorParallel};
    config.automatic_planning.device_counts={{DeviceType::ROCm,2}};config.kv_cache_precision="fp32";
    const auto selected=AutomaticOrchestrationPlanner::select(config,source,observed,memory(),{64,128},
        [](const auto&,const auto& work){return OrchestrationCostEstimate(work,1,.01,"synthetic equal admitted service");});
    EXPECT_GT(selected.counts().capacity_rejected,0u);
    std::set<int> ids;for(const auto& p:selected.candidate().devicePlans()) ids.insert(p.device.ordinal);
    EXPECT_EQ(ids,(std::set<int>{91,7}));
}

TEST(ObservedPeerGrouping, UUIDAliasesRejectContradictoryOriginalISAAndPCIAncestry)
{
    for(bool isa:{false,true}) {
        ClusterInventory observed;observed.world_size=2;
        observed.ranks.push_back(peerIslands(2));observed.ranks.push_back(observed.ranks.front());
        observed.ranks[1].rank=1;observed.ranks[1].local_rank=1;
        EXPECT_NO_THROW(observed.buildNodeAggregations());
        if(isa) observed.ranks[1].gpus[0].native_isa="gfx1100";
        else observed.ranks[1].gpus[0].pcie_upstream_bdfs={"0000:ff:00.0"};
        EXPECT_THROW(observed.buildNodeAggregations(),std::invalid_argument);
    }
}

/** @test Stage eligibility retains the uniform API's original island and UUID tie order. */
TEST(ObservedPeerGrouping, StageEligibilityRetainsUniformAPIResultsAndStableIslandTies)
{
    for(const bool missing:{false,true}) {
        auto rank=peerIslands();
        if(missing) {rank.p2p_rocm.clear();for(auto& gpu:rank.gpus)gpu.pcie_upstream_bdfs.clear();}
        std::vector<DeviceId> eligible;
        for(const auto& gpu:rank.gpus)eligible.emplace_back(gpu.type,gpu.local_device_id);
        for(const auto sizes:{std::vector<std::size_t>{4,4},std::vector<std::size_t>{1,2},
                std::vector<std::size_t>{2,1},std::vector<std::size_t>{3,5},std::vector<std::size_t>{1,3,4}}) {
            const ObservedPeerGrouping grouping(rank);
            const auto legacy=grouping.select(eligible,sizes);
            std::vector<ObservedPeerStageRequest> request;
            for(const auto size:sizes)request.push_back({size,eligible});
            const auto constrained=grouping.select(request);
            ASSERT_TRUE(constrained);EXPECT_EQ(*constrained,legacy);
            if(!missing && sizes==std::vector<std::size_t>{4,4})
                EXPECT_EQ(groupOrdinals(*constrained),(std::set<std::set<int>>{{91,53,87,61},{7,2,11,3}}));
            std::reverse(eligible.begin(),eligible.end());
            // request spans still refer to the same stable vector allocation.
            const auto reordered=grouping.select(request);
            ASSERT_TRUE(reordered);EXPECT_EQ(*reordered,legacy);
        }
    }
}

/** @test The one canonical search distinguishes infeasibility from malformed metadata and prunes only impossible suffixes. */
TEST(ObservedPeerGrouping, StageEligibilityKeepsTypedInfeasibilityAndExactLargeCohorts)
{
    auto rank=peerIslands(4);
    const DeviceId a=DeviceId::rocm(91),b=DeviceId::rocm(7),c=DeviceId::rocm(53);
    const std::array wide{a,b};const std::array only_a{a};const std::array other{b,c};
    const ObservedPeerGrouping grouping(rank);
    const std::array request{ObservedPeerStageRequest{1,wide},ObservedPeerStageRequest{1,only_a}};
    const auto selected=grouping.select(request);
    ASSERT_TRUE(selected);EXPECT_EQ(*selected,(std::vector<std::vector<DeviceId>>{{b},{a}}));
    const std::array impossible{ObservedPeerStageRequest{1,only_a},ObservedPeerStageRequest{1,only_a},
        ObservedPeerStageRequest{1,other}};
    EXPECT_FALSE(grouping.select(impossible));
    const std::array zero{ObservedPeerStageRequest{0,only_a}};
    EXPECT_THROW((void)grouping.select(zero),std::invalid_argument);
    const std::array overflow{ObservedPeerStageRequest{std::numeric_limits<std::size_t>::max(),only_a},
        ObservedPeerStageRequest{1,wide}};
    EXPECT_THROW((void)grouping.select(overflow),std::invalid_argument);
    const std::array duplicate{a,a};const std::array invalid{ObservedPeerStageRequest{1,duplicate}};
    EXPECT_THROW((void)grouping.select(invalid),std::invalid_argument);
    const std::array absent{DeviceId::rocm(666)};const std::array unknown{ObservedPeerStageRequest{1,absent}};
    EXPECT_THROW((void)grouping.select(unknown),std::invalid_argument);
    rank.gpus[1].uuid=rank.gpus[0].uuid;
    EXPECT_THROW((void)grouping.select(request),std::invalid_argument);

    // One exact 65-device stage has a single feasible subset. Remaining-suffix
    // pruning avoids enumerating its exponentially many impossible omissions;
    // no participant mask, heuristic or small-count specialization is needed.
    auto large=peerIslands(1);large.gpus.clear();large.p2p_rocm.clear();large.p2p_rocm_count=65;
    std::vector<DeviceId> eligible;
    for(int index=0;index<65;++index) {
        DeviceInfo gpu;gpu.type=DeviceType::ROCm;gpu.local_device_id=1000+index*3;
        gpu.numa_node=large.cpu.numa_node;gpu.uuid="large-original-"+std::to_string(index);
        eligible.emplace_back(gpu.type,gpu.local_device_id);large.gpus.push_back(std::move(gpu));
    }
    const ObservedPeerGrouping exact(large);
    const std::array large_request{ObservedPeerStageRequest{65,eligible}};
    const auto large_selected=exact.select(large_request);
    ASSERT_TRUE(large_selected);ASSERT_EQ(large_selected->size(),1u);EXPECT_EQ(large_selected->front().size(),65u);
    EXPECT_EQ(exact.select(eligible,std::array<std::size_t,1>{65}),*large_selected);
}
