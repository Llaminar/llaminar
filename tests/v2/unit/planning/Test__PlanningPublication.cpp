/**
 * @file Test__PlanningPublication.cpp
 * @brief Device-free contracts for the shared metadata/selection publication.
 *
 * Null-context tests deliberately never initialize MPI or a backend. They
 * exercise the exact production codec, mandatory selection identity and local
 * failure terminals. Real asymmetric peers belong to the MPI integration gate.
 */
#include "planning/PlanningPublication.h"
#include "config/OrchestrationConfigDocument.h"
#include <gtest/gtest.h>
#include <stdexcept>

using namespace llaminar2;

namespace
{
    /** @return A complete apply request with non-default inference policy. */
    OrchestrationConfig selection(DeviceType backend)
    {
        OrchestrationConfig config;
        config.model_path = "/model directory/weights.gguf";
        config.planning_mode = OrchestrationPlanningMode::Apply;
        config.device_for_this_rank = GlobalDeviceAddress{"host", 0, backend, 0};
        config.execution_rank_selection = ExecutionRankSelection({0});
        config.mpi_procs = 1;
        config.max_seq_len = 4096;
        config.kv_cache_precision = "fp32";
        config.mtp.enabled = true;
        config.mtp.graph_capacity_draft_tokens = 15;
        config.mtp.depth_policy.mode = MTPDepthPolicyMode::Dynamic;
        config.mtp.depth_policy.max_depth = 15;
        config.prefill_max_bucket_size = 64;
        return config;
    }
}

TEST(PlanningPublication, ProcessLocalArtifactHasOneProducerAndOneDecoder)
{
    for (const auto artifact : {PlanningArtifact::AutomaticRequest, PlanningArtifact::ModelMetadata, PlanningArtifact::SelectedOrchestration,
                               PlanningArtifact::TransferSamples, PlanningArtifact::ExpertSample,
                               PlanningArtifact::MatrixSample})
    {
        int produced = 0, accepted = 0;
        const std::vector<uint8_t> expected{0, 255, 1, 0, 2};
        exchangePlanningArtifact(nullptr, artifact, [&] { ++produced; return expected; },
            [&](auto bytes) {
                ++accepted;
                EXPECT_EQ(std::vector<uint8_t>(bytes.begin(), bytes.end()), expected);
            });
        EXPECT_EQ(produced, 1);
        EXPECT_EQ(accepted, 1);
    }
}

TEST(PlanningPublication, LocalCostCompletionAuthenticatesFailureBeforeSelection)
{
    size_t accepted = 0;
    EXPECT_NO_THROW(acceptPlanningCostPreparation({}, [&] { ++accepted; }));
    EXPECT_EQ(accepted, 1u);
    EXPECT_THROW(acceptPlanningCostPreparation({}, {}), std::runtime_error);
    EXPECT_THROW(acceptPlanningCostPreparation({}, [] {
        throw std::runtime_error("Cost closure could not be constructed");
    }), std::runtime_error);
}

TEST(PlanningPublication, InvalidCallbacksPayloadOrArtifactFailBeforeDecoding)
{
    int decoded = 0, produced = 0;
    const auto producer = [&] { ++produced; return std::vector<uint8_t>{1}; };
    const auto decoder = [&](auto) { ++decoded; };
    EXPECT_THROW(exchangePlanningArtifact(nullptr, PlanningArtifact::ModelMetadata, {}, decoder), std::runtime_error);
    EXPECT_THROW(exchangePlanningArtifact(nullptr, PlanningArtifact::ModelMetadata, producer, {}), std::runtime_error);
    EXPECT_THROW(exchangePlanningArtifact(nullptr, static_cast<PlanningArtifact>(99), producer, decoder), std::runtime_error);
    EXPECT_EQ(produced, 0);
    EXPECT_THROW(exchangePlanningArtifact(nullptr, PlanningArtifact::ModelMetadata,
        [] { return std::vector<uint8_t>{}; }, decoder), std::runtime_error);
    EXPECT_THROW(exchangePlanningArtifact(nullptr, PlanningArtifact::ModelMetadata,
        []() -> std::vector<uint8_t> { throw std::runtime_error("read failed"); }, decoder), std::runtime_error);
    EXPECT_EQ(decoded, 0);
    EXPECT_THROW(exchangePlanningArtifact(nullptr, PlanningArtifact::ModelMetadata, producer,
        [](auto) { throw std::runtime_error("decode failed"); }), std::runtime_error);
}

TEST(PlanningPublication, SelectedConfigurationRetainsExactPolicyForEveryBackend)
{
    for (auto backend : {DeviceType::CPU, DeviceType::CUDA, DeviceType::ROCm})
    {
        const auto expected = selection(backend);
        int calls = 0;
        const auto actual = exchangeSelectedOrchestration(nullptr, [&] { ++calls; return expected; });
        EXPECT_EQ(calls, 1);
        EXPECT_EQ(serializeOrchestrationConfig(actual), serializeOrchestrationConfig(expected));
    }
}

TEST(PlanningPublication, UnresolvedMissingOrInvalidSelectionCannotBeApplied)
{
    EXPECT_THROW((void)exchangeSelectedOrchestration(nullptr, {}), std::runtime_error);
    for (int defect = 0; defect < 6; ++defect)
    {
        auto config = selection(DeviceType::CPU);
        switch (defect)
        {
        case 0: config.execution_rank_selection.reset(); break;
        case 1: config.model_path.clear(); break;
        case 2: config.mpi_procs = 2; break;
        case 3: config.execution_rank_selection = ExecutionRankSelection({1}); break;
        case 4: config.planning_mode = OrchestrationPlanningMode::Automatic; break;
        case 5: config.max_seq_len = -1; break;
        }
        SCOPED_TRACE(defect);
        EXPECT_THROW((void)exchangeSelectedOrchestration(nullptr, [&] { return config; }), std::runtime_error);
    }
}

/** @test The one-rank sample transaction preserves byte envelopes and callback ownership. */
TEST(PlanningPublication, LocalSamplesUseTheSameScatterExecuteGatherContract)
{
    int planned = 0, executed = 0, accepted = 0;
    exchangePlanningSamples({}, [&](int count) {
        ++planned;
        EXPECT_EQ(count, 1);
        return std::vector<std::vector<uint8_t>>{{0, 127, 255}};
    }, [&](auto request) {
        ++executed;
        EXPECT_EQ(std::vector<uint8_t>(request.begin(), request.end()), (std::vector<uint8_t>{0, 127, 255}));
        return std::vector<uint8_t>{19, 0, 41, 255};
    }, [&](auto samples) {
        ++accepted;
        ASSERT_EQ(samples.size(), 1);
        EXPECT_EQ(samples[0].discovery_rank, 0);
        EXPECT_EQ(std::vector<uint8_t>(samples[0].bytes.begin(), samples[0].bytes.end()),
            (std::vector<uint8_t>{19, 0, 41, 255}));
    });
    EXPECT_EQ(planned, 1);
    EXPECT_EQ(executed, 1);
    EXPECT_EQ(accepted, 1);
}

/** @test No failed/missing preparation or measurement can supply a partial score. */
TEST(PlanningPublication, LocalSampleFailuresStopAtTheirOwnPhase)
{
    for (int defect = 0; defect != 10; ++defect)
    {
        SCOPED_TRACE(defect);
        int executed = 0, accepted = 0;
        std::function<std::vector<std::vector<uint8_t>>(int)> plan = [&](int) {
            if (defect == 1) throw std::runtime_error("description failed");
            if (defect == 2) return std::vector<std::vector<uint8_t>>{};
            if (defect == 3) return std::vector<std::vector<uint8_t>>{{}};
            if (defect == 4) return std::vector<std::vector<uint8_t>>{{1}, {2}};
            return std::vector<std::vector<uint8_t>>{{1}};
        };
        std::function<std::vector<uint8_t>(std::span<const uint8_t>)> run = [&](auto) {
            ++executed;
            if (defect == 6) throw std::runtime_error("native execution failed");
            return defect == 7 ? std::vector<uint8_t>{} : std::vector<uint8_t>{2};
        };
        std::function<void(std::span<const RankPlanningSample>)> accept = [&](auto) {
            ++accepted;
            if (defect == 9) throw std::runtime_error("sample identity mismatch");
        };
        if (defect == 0) plan = {};
        if (defect == 5) run = {};
        if (defect == 8) accept = {};
        EXPECT_THROW(exchangePlanningSamples({}, plan, run, accept), std::runtime_error);
        EXPECT_EQ(executed, defect == 6 || defect == 7 || defect == 9 ? 1 : 0);
        EXPECT_EQ(accepted, defect == 9 ? 1 : 0);
    }
}
