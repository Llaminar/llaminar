/**
 * @file Test__InferenceRunnerFactory_MultiDevice.cpp
 * @brief Unit tests for MultiDevice factory functions in InferenceRunnerFactory
 *
 * Tests the createRankOrchestrator and createTestableRankOrchestrator
 * factory functions with various configurations.
 *
 * @author David Sanftenberg
 * @date January 2026
 */

#include <gtest/gtest.h>
#include <gmock/gmock.h>

#include "execution/factory/InferenceRunnerFactory.h"
#include "execution/local_execution/orchestrators/RankOrchestrator.h"
#include "execution/local_execution/orchestrators/DeviceGraphOrchestrator.h"
#include "execution/local_execution/orchestrators/IRankOrchestrator.h"
#include "execution/moe/MoERebalanceController.h"
#include "execution/moe/MoEExpertOverlayRuntimePlan.h"
#include "execution/moe/MoERoutedExpertPlacementPlanner.h"
#include "collective/ILocalTPContext.h"
#include "collective/IGlobalTPContext.h"
#include "backends/GlobalDeviceAddress.h"
#include "models/GraphTypes.h"
#include "mocks/MockModelContext.h"

#include <fstream>
#include <sstream>
#include <string>

using namespace llaminar2;
using namespace llaminar2::test;
using namespace testing;

namespace
{
    /**
     * @brief Read a repository source file for a factory source-contract test.
     *
     * Some runner-factory ownership bugs only appear when a real GGUF model,
     * nested TP-in-PP stage config, and decode-replicated dense sidecar are all
     * present. The source-contract tests below guard the precise factory wiring
     * rule without forcing every unit-test run to materialize that full stack.
     *
     * @param path Repository-relative path to read from the CTest working tree.
     * @return File contents, or an empty string when the file cannot be read.
     */
    std::string readFactorySourceFile(const std::string &path)
    {
        std::ifstream input(path);
        if (!input.good())
            return {};

        std::ostringstream buffer;
        buffer << input.rdbuf();
        return buffer.str();
    }

    /**
     * @brief Count non-overlapping string occurrences in source text.
     *
     * @param haystack Source text to inspect.
     * @param needle Text to count.
     * @return Number of non-overlapping matches.
     */
    size_t countFactorySourceOccurrences(
        const std::string &haystack,
        const std::string &needle)
    {
        size_t count = 0;
        size_t pos = 0;
        while ((pos = haystack.find(needle, pos)) != std::string::npos)
        {
            ++count;
            pos += needle.size();
        }
        return count;
    }

    // =============================================================================
    // Mock LocalTPContext for factory tests
    // =============================================================================

    /**
     * @brief Simple mock for ILocalTPContext used in factory tests
     */
    class MockLocalTPContext : public ILocalTPContext
    {
    public:
        MockLocalTPContext(
            std::vector<GlobalDeviceAddress> devices,
            std::vector<float> weights,
            CollectiveBackendType backend = CollectiveBackendType::HOST)
            : devices_(std::move(devices)), weights_(std::move(weights)), backend_(backend)
        {
        }

        const std::vector<GlobalDeviceAddress> &devices() const override { return devices_; }
        const std::vector<float> &weights() const override { return weights_; }
        CollectiveBackendType backend() const override { return backend_; }
        int degree() const override { return static_cast<int>(devices_.size()); }
        int myIndex() const override { return 0; }

        bool allreduce(TensorBase * /*tensor*/) override { return true; }
        bool allreduce(TensorBase *tensor, const std::string & /*stage_name*/, size_t /*count*/ = 0) override { return allreduce(tensor); }
        bool allreduce(const TensorBase * /*input*/, TensorBase * /*output*/) override { return true; }
        bool allgather(const TensorBase * /*local_shard*/, TensorBase * /*global_tensor*/) override { return true; }
        bool reduceScatter(const TensorBase * /*input*/, TensorBase * /*output_shard*/) override { return true; }

        void synchronize() override {}

        int indexForDevice(const GlobalDeviceAddress &device) const override
        {
            for (size_t i = 0; i < devices_.size(); ++i)
            {
                if (devices_[i] == device)
                {
                    return static_cast<int>(i);
                }
            }
            return -1;
        }

        const GlobalDeviceAddress &deviceAt(int index) const override
        {
            return devices_.at(static_cast<size_t>(index));
        }

        float weightForDevice(const GlobalDeviceAddress &device) const override
        {
            int idx = indexForDevice(device);
            return (idx >= 0) ? weights_[static_cast<size_t>(idx)] : 0.0f;
        }

        int headsForDevice(const GlobalDeviceAddress &device, int total_heads) const override
        {
            float w = weightForDevice(device);
            return static_cast<int>(w * static_cast<float>(total_heads) + 0.5f);
        }

        std::pair<int, int> rowRangeForDevice(
            const GlobalDeviceAddress &device, int total_rows) const override
        {
            int idx = indexForDevice(device);
            if (idx < 0)
                return {0, 0};

            float cumulative = 0.0f;
            for (int i = 0; i < idx; ++i)
            {
                cumulative += weights_[static_cast<size_t>(i)];
            }
            int start = static_cast<int>(cumulative * static_cast<float>(total_rows));
            int end = static_cast<int>((cumulative + weights_[static_cast<size_t>(idx)]) * static_cast<float>(total_rows));
            return {start, end};
        }

        std::pair<int, int> colRangeForDevice(
            const GlobalDeviceAddress &device, int total_cols) const override
        {
            return rowRangeForDevice(device, total_cols);
        }

        bool gatherFromDevices(
            const std::vector<const TensorBase *> & /*shards*/,
            TensorBase * /*output*/) override
        {
            return true;
        }

        // BAR Registry (no-ops for tests)
        void registerBARBackedOutput(
            const std::string & /*stage_name*/,
            const GlobalDeviceAddress & /*device*/,
            TensorBase * /*tensor*/) override
        {
        }
        bool hasBARBackedOutputs(const std::string & /*stage_name*/) const override { return false; }
        void clearBARBackedOutputs() override {}
        bool reserveTempBufferBytes(size_t /*bytes*/) override { return true; }

        // Broadcast (no-op)
        bool broadcast(TensorBase * /*tensor*/, int /*source_device_index*/ = 0) override { return true; }

        void requestAbort() override {}
        bool isAbortRequested() const override { return false; }

    private:
        std::vector<GlobalDeviceAddress> devices_;
        std::vector<float> weights_;
        CollectiveBackendType backend_ = CollectiveBackendType::HOST;
    };

    class FakeGlobalTPContext : public IGlobalTPContext
    {
    public:
        FakeGlobalTPContext(int domain_id, int my_index, int degree)
            : domain_id_(domain_id), my_index_(my_index), degree_(degree)
        {
            for (int rank = 0; rank < degree_; ++rank)
                world_ranks_.push_back(rank);
        }

        int degree() const override { return degree_; }
        int myIndex() const override { return my_index_; }
        CollectiveBackendType backend() const override { return CollectiveBackendType::MPI; }
        MPI_Comm communicator() const override { return MPI_COMM_SELF; }
        int domainId() const override { return domain_id_; }
        const std::vector<int> &worldRanks() const override { return world_ranks_; }
        GlobalDeviceAddress localDevice() const override { return GlobalDeviceAddress::cpu(my_index_); }
        void barrier() const override {}
        bool allreduce(TensorBase *) override { return false; }
        bool broadcast(TensorBase *, int = 0) override { return false; }
        bool allgather(const TensorBase *, TensorBase *) override { return false; }
        bool send(const TensorBase *, int) override { return false; }
        bool recv(TensorBase *, int) override { return false; }

    private:
        int domain_id_ = 0;
        int my_index_ = 0;
        int degree_ = 1;
        std::vector<int> world_ranks_;
    };

    constexpr int kMoELayers = 3;
    constexpr int kMoEExperts = 6;
    constexpr int kMoEDModel = 16;
    constexpr int kMoEIntermediate = 8;
    constexpr size_t kF32RoutedExpertBytes =
        3u * static_cast<size_t>(kMoEDModel) * static_cast<size_t>(kMoEIntermediate) * sizeof(float);

    std::shared_ptr<MockModelContext> makeMoEModelContext()
    {
        auto model_ctx = MockModelContextBuilder()
                             .setArchitecture("qwen3moe")
                             .setBlockCount(kMoELayers)
                             .setEmbeddingLength(kMoEDModel)
                             .setHeadCount(4)
                             .setHeadCountKV(2)
                             .setVocabSize(128)
                             .setContextLength(256)
                             .setFeedForwardLength(kMoEIntermediate)
                             .build();

        model_ctx->mockLoader().setIntParam("qwen3moe.expert_count", kMoEExperts);
        model_ctx->mockLoader().setIntParam("qwen3moe.expert_feed_forward_length", kMoEIntermediate);
        model_ctx->mockLoader().setIntParam("qwen3moe.expert_shared_count", 1);
        return model_ctx;
    }

    RoutedExpertDomain overlayDomain(
        const std::string &name,
        GlobalDeviceAddress participant)
    {
        RoutedExpertDomain domain;
        domain.name = name;
        domain.scope = ExecutionDomainScope::SINGLE;
        domain.backend = CollectiveBackendType::AUTO;
        domain.participants = {std::move(participant)};
        domain.routed_compute_policy = RoutedExpertComputePolicy::Apportioned;
        return domain;
    }

    RoutedExpertTier overlayTier(
        const std::string &name,
        const std::string &domain,
        int priority,
        bool fallback = false)
    {
        RoutedExpertTier tier;
        tier.name = name;
        tier.domain = domain;
        tier.priority = priority;
        tier.fallback = fallback;
        return tier;
    }

    std::shared_ptr<MoERoutedExpertPlacementPlan> makeRequestedOverlayPlan(
        RoutedExpertResidencyPolicy policy = RoutedExpertResidencyPolicy::StaticById)
    {
        auto plan = std::make_shared<MoERoutedExpertPlacementPlan>();
        plan->enabled = true;
        plan->topology = RoutedExpertPlacementTopology::TieredOverlay;
        plan->continuation_domain = "gpu_hot";
        plan->shared_expert_domain = "gpu_hot";
        plan->residency_policy = policy;
        plan->domains = {
            overlayDomain("gpu_hot", GlobalDeviceAddress::cuda(0)),
            overlayDomain("cpu_cold", GlobalDeviceAddress::cpu()),
        };
        plan->routed_tiers = {
            overlayTier("hot", "gpu_hot", 0),
            overlayTier("cold", "cpu_cold", 1, true),
        };
        plan->routed_tiers[0].max_experts_per_layer = kMoEExperts;
        plan->routed_tiers[0].memory_budget_bytes = 2u * kF32RoutedExpertBytes;
        return plan;
    }

    std::shared_ptr<MoERoutedExpertPlacementPlan> makeActiveRocmLocalTPOverlayPlan()
    {
        auto plan = std::make_shared<MoERoutedExpertPlacementPlan>();
        plan->enabled = true;
        plan->topology = RoutedExpertPlacementTopology::TieredOverlay;
        plan->continuation_domain = "rocm_hot";
        plan->shared_expert_domain = "rocm_hot";
        plan->residency_policy = RoutedExpertResidencyPolicy::ExplicitMasks;

        RoutedExpertDomain rocm_hot;
        rocm_hot.name = "rocm_hot";
        rocm_hot.scope = ExecutionDomainScope::LOCAL;
        rocm_hot.backend = CollectiveBackendType::RCCL;
        rocm_hot.routed_compute_policy = RoutedExpertComputePolicy::TensorSharded;
        rocm_hot.participants = {
            GlobalDeviceAddress::rocm(0),
            GlobalDeviceAddress::rocm(1),
        };

        RoutedExpertDomain cpu_cold;
        cpu_cold.name = "cpu_cold";
        cpu_cold.scope = ExecutionDomainScope::NODE_LOCAL;
        cpu_cold.backend = CollectiveBackendType::HOST;
        cpu_cold.routed_compute_policy = RoutedExpertComputePolicy::Apportioned;
        cpu_cold.participants = {GlobalDeviceAddress::cpu()};

        plan->domains = {rocm_hot, cpu_cold};
        plan->routed_tiers = {
            overlayTier("hot", "rocm_hot", 0),
            overlayTier("cold", "cpu_cold", 1, true),
        };
        plan->placements = {
            RoutedExpertLayerPlacement{.layer = 0, .routed_expert_tier = {0, 0, 1, 1, 1, 1}},
        };
        return plan;
    }

    std::shared_ptr<MoERoutedExpertPlacementPlan> makeActiveRocmLocalTPReplicatedOverlayPlan()
    {
        auto plan = makeActiveRocmLocalTPOverlayPlan();
        plan->domains[0].routed_compute_policy = RoutedExpertComputePolicy::Apportioned;
        return plan;
    }

    std::shared_ptr<MoERoutedExpertPlacementPlan> makeActiveCudaLocalTPReplicatedOverlayPlan()
    {
        auto plan = std::make_shared<MoERoutedExpertPlacementPlan>();
        plan->enabled = true;
        plan->topology = RoutedExpertPlacementTopology::TieredOverlay;
        plan->continuation_domain = "cuda_hot";
        plan->shared_expert_domain = "cuda_hot";
        plan->residency_policy = RoutedExpertResidencyPolicy::ExplicitMasks;

        RoutedExpertDomain cuda_hot;
        cuda_hot.name = "cuda_hot";
        cuda_hot.scope = ExecutionDomainScope::LOCAL;
        cuda_hot.backend = CollectiveBackendType::NCCL;
        cuda_hot.routed_compute_policy = RoutedExpertComputePolicy::Apportioned;
        cuda_hot.participants = {
            GlobalDeviceAddress::cuda(0),
            GlobalDeviceAddress::cuda(1),
        };

        plan->domains = {cuda_hot};
        plan->routed_tiers = {
            overlayTier("hot", "cuda_hot", 0),
        };
        return plan;
    }

    // =============================================================================
    // Test Fixture
    // =============================================================================

    class Test__InferenceRunnerFactory_MultiDevice : public ::testing::Test
    {
    protected:
        void SetUp() override
        {
            model_ctx_ = MockModelContextBuilder()
                             .usePreset(ModelPreset::MINIMAL)
                             .build();
        }

        std::shared_ptr<MockModelContext> model_ctx_;
    };

    // =============================================================================
    // createRankOrchestrator Tests
    // =============================================================================

    TEST_F(Test__InferenceRunnerFactory_MultiDevice, NullModelContextReturnsNull)
    {
        // Create valid TP context
        auto tp_ctx = std::make_unique<MockLocalTPContext>(
            std::vector<GlobalDeviceAddress>{GlobalDeviceAddress::cpu()},
            std::vector<float>{1.0f});

        RankOrchestrator::Config config;
        config.devices = tp_ctx->devices();

        auto result = createRankOrchestrator(nullptr, std::move(tp_ctx), config);
        EXPECT_EQ(result, nullptr);
    }

    TEST_F(Test__InferenceRunnerFactory_MultiDevice, NullTPContextReturnsNull)
    {
        RankOrchestrator::Config config;
        config.devices = {GlobalDeviceAddress::cpu()};

        auto result = createRankOrchestrator(model_ctx_, nullptr, config);
        EXPECT_EQ(result, nullptr);
    }

    TEST_F(Test__InferenceRunnerFactory_MultiDevice, EmptyDevicesReturnsNull)
    {
        auto tp_ctx = std::make_unique<MockLocalTPContext>(
            std::vector<GlobalDeviceAddress>{},
            std::vector<float>{});

        RankOrchestrator::Config config;
        // config.devices is empty - should fail validation

        auto result = createRankOrchestrator(model_ctx_, std::move(tp_ctx), config);
        EXPECT_EQ(result, nullptr);
    }

    TEST_F(Test__InferenceRunnerFactory_MultiDevice, InvalidWeightsReturnsNull)
    {
        auto tp_ctx = std::make_unique<MockLocalTPContext>(
            std::vector<GlobalDeviceAddress>{GlobalDeviceAddress::cpu(), GlobalDeviceAddress::cpu()},
            std::vector<float>{0.3f, 0.3f}); // Sum != 1.0

        RankOrchestrator::Config config;
        config.devices = tp_ctx->devices();
        config.weights = {0.3f, 0.3f}; // Invalid - doesn't sum to 1.0

        auto result = createRankOrchestrator(model_ctx_, std::move(tp_ctx), config);
        EXPECT_EQ(result, nullptr);
    }

    // =============================================================================
    // createTestableRankOrchestrator Tests
    // =============================================================================

    TEST_F(Test__InferenceRunnerFactory_MultiDevice, TestableWithNullModelCtxReturnsNull)
    {
        std::vector<std::unique_ptr<IInferenceRunner>> runners;
        // Can't create real runners without model - this tests null check

        RankOrchestrator::Config config;
        config.devices = {GlobalDeviceAddress::cpu()};

        auto result = createTestableRankOrchestrator(
            nullptr, std::move(runners), nullptr, config);
        EXPECT_EQ(result, nullptr);
    }

    TEST_F(Test__InferenceRunnerFactory_MultiDevice, TestableWithEmptyRunnersReturnsNull)
    {
        std::vector<std::unique_ptr<IInferenceRunner>> empty_runners;

        RankOrchestrator::Config config;
        config.devices = {GlobalDeviceAddress::cpu()};

        auto result = createTestableRankOrchestrator(
            model_ctx_, std::move(empty_runners), nullptr, config);
        EXPECT_EQ(result, nullptr);
    }

    /**
     * @brief Replicated dense decode sidecar plans must preserve PP ownership.
     *
     * A nested TP-in-PP runner has two dense weight views: the primary frozen
     * stage weights and the decode-replicated dense sidecar. Both views must be
     * filtered by the same FactoryPPStageConfig. Otherwise a terminal stage that
     * does not own embeddings can reject its decode sidecar for missing
     * token_embd.weight, while a non-terminal stage can accidentally materialize
     * LM-head bindings it should never own.
     */
    TEST(Test__InferenceRunnerFactory_SourceContract,
         DecodeReplicatedDensePlansUsePPStageFilter)
    {
        const std::string source =
            readFactorySourceFile("src/v2/execution/factory/InferenceRunnerFactory.cpp");
        ASSERT_FALSE(source.empty());

        EXPECT_NE(source.find("[InferenceRunner] PP stage decode replicated dense"),
                  std::string::npos)
            << "explicit PP materialization must build a decode-replicated dense sidecar";
        EXPECT_NE(source.find("graph_config.tp_config.get(),\n"
                              "                    &pp_cfg,\n"
                              "                    SingleDeviceWeightPlanOptions"),
                  std::string::npos)
            << "explicit PP sidecar plans must use the stage's FactoryPPStageConfig";
        EXPECT_GE(countFactorySourceOccurrences(
                      source,
                      "const FactoryPPStageConfig *decode_pp_config ="),
                  2u)
            << "generic decode sidecar plan sites must derive their PP filter from runner config";
        EXPECT_GE(countFactorySourceOccurrences(
                      source,
                      "requiresTerminalMTPSidecarEmbedding(graph_config,"),
                  3u)
            << "decode-replicated dense sidecar plans must carry terminal MTP embedding ownership "
               "instead of treating token_embd.weight as a full-stage embedding fallback.";
        EXPECT_NE(source.find("graph_config.tp_config.get(),\n"
                              "                decode_pp_config,\n"
                              "                SingleDeviceWeightPlanOptions"),
                  std::string::npos)
            << "concrete runner sidecar plans must pass the runner's optional PP filter";
        EXPECT_NE(source.find("graph_config.tp_config.get(),\n"
                              "                            decode_pp_config,\n"
                              "                            SingleDeviceWeightPlanOptions"),
                  std::string::npos)
            << "testable LocalTP sidecar plans must pass the runner's optional PP filter";
    }

    /**
     * @brief Replicated dense decode plans must keep MTP verifier sidecar weights.
     *
     * Base-layer routed experts are excluded from the dense decode subset because
     * ExpertOverlay owns their residency separately. Qwen3.6 MTP is different:
     * its trailing nextn block is the verifier path itself, so even nextn tensors
     * with ffn_*_exps names must remain in the replicated decode weight plan.
     */
    TEST(Test__InferenceRunnerFactory_SourceContract,
         DecodeReplicatedDensePlansRetainNextNMTPSidecarWeights)
    {
        const std::string source =
            readFactorySourceFile("src/v2/execution/factory/InferenceRunnerFactory.cpp");
        ASSERT_FALSE(source.empty());

        EXPECT_NE(source.find("weight_name.find(\".nextn.\")"),
                  std::string::npos)
            << "phase-split dense decode must keep Qwen3.6 nextn verifier weights";
        EXPECT_NE(source.find("weight_name.rfind(\"mtp.\", 0)"),
                  std::string::npos)
            << "generic mtp.layers verifier weights must also remain in the dense decode plan";
        EXPECT_NE(source.find("discoverDenseDecodeMTPSourceLayers"),
                  std::string::npos)
            << "Qwen3.6 nextn source-layer discovery must keep normal-named sidecar attention/expert weights";
        EXPECT_NE(source.find("inferWeightLayer(weight_name)"),
                  std::string::npos)
            << "dense decode filtering must recognize the full trailing nextn layer, not only names containing nextn";
        EXPECT_NE(source.find("grouped MTP rows remain mathematically identical to serial decode"),
                  std::string::npos)
            << "the source contract should document the parity reason for keeping MTP sidecar weights";
    }

    TEST(Test__InferenceRunnerFactory_MoEOverlayPlanning, PlansMissingPlacementsFromModelMetadata)
    {
        auto model_ctx = makeMoEModelContext();
        auto requested_plan = makeRequestedOverlayPlan();

        InferenceRunnerConfig config;
        config.moe_routed_expert_plan = requested_plan;

        auto resolved_plan = resolveMoERoutedExpertPlacementPlanForModel(*model_ctx, config);

        ASSERT_NE(resolved_plan, nullptr);
        EXPECT_NE(resolved_plan.get(), requested_plan.get());
        EXPECT_TRUE(requested_plan->placements.empty());
        ASSERT_EQ(resolved_plan->placements.size(), static_cast<size_t>(kMoELayers));
        for (int layer = 0; layer < kMoELayers; ++layer)
        {
            const auto &placement = resolved_plan->placements[static_cast<size_t>(layer)];
            EXPECT_EQ(placement.layer, layer);
            EXPECT_EQ(placement.routed_expert_tier,
                      (std::vector<int>{0, 0, 1, 1, 1, 1}));
        }
        ASSERT_EQ(resolved_plan->routed_tiers.size(), 2u);
        EXPECT_EQ(resolved_plan->routed_tiers[1].domain, "cpu_cold");
        EXPECT_TRUE(resolved_plan->routed_tiers[1].fallback);
    }

    TEST(Test__InferenceRunnerFactory_MoEOverlayPlanning, PreservesExplicitPlacements)
    {
        auto model_ctx = makeMoEModelContext();
        auto explicit_plan = makeRequestedOverlayPlan(RoutedExpertResidencyPolicy::ExplicitMasks);
        explicit_plan->placements = {
            RoutedExpertLayerPlacement{.layer = 0, .routed_expert_tier = {0, 1, 0, 1, 0, 1}},
            RoutedExpertLayerPlacement{.layer = 1, .routed_expert_tier = {1, 0, 1, 0, 1, 0}},
            RoutedExpertLayerPlacement{.layer = 2, .routed_expert_tier = {0, 0, 1, 1, 0, 1}},
        };

        InferenceRunnerConfig config;
        config.moe_routed_expert_plan = explicit_plan;

        auto resolved_plan = resolveMoERoutedExpertPlacementPlanForModel(*model_ctx, config);

        EXPECT_EQ(resolved_plan, explicit_plan);
        ASSERT_EQ(resolved_plan->placements.size(), 3u);
        EXPECT_EQ(resolved_plan->placements[0].routed_expert_tier,
                  (std::vector<int>{0, 1, 0, 1, 0, 1}));
        EXPECT_EQ(resolved_plan->placements[1].routed_expert_tier,
                  (std::vector<int>{1, 0, 1, 0, 1, 0}));
        EXPECT_EQ(resolved_plan->placements[2].routed_expert_tier,
                  (std::vector<int>{0, 0, 1, 1, 0, 1}));
    }

    TEST(Test__InferenceRunnerFactory_MoEOverlayPlanning, PlanningErrorsSurfaceBeforeGraphExecution)
    {
        auto model_ctx = makeMoEModelContext();
        auto invalid_plan = makeRequestedOverlayPlan(RoutedExpertResidencyPolicy::Disabled);

        InferenceRunnerConfig config;
        config.moe_routed_expert_plan = invalid_plan;

        try
        {
            (void)resolveMoERoutedExpertPlacementPlanForModel(*model_ctx, config);
            FAIL() << "Expected overlay planning to fail";
        }
        catch (const std::invalid_argument &e)
        {
            EXPECT_THAT(std::string(e.what()), HasSubstr("Disabled residency policy"));
        }
    }

    TEST(Test__InferenceRunnerFactory_MoEOverlayPlanning, OverlayExecutionDeviceRejectsNonContinuationParticipant)
    {
        GraphConfig graph_config;
        graph_config.moe.routed_expert_plan = makeActiveRocmLocalTPOverlayPlan();

        EXPECT_THROW(
            (void)resolveMoEExpertOverlayExecutionDeviceForGraph(
                graph_config,
                nullptr,
                DeviceId::cpu(),
                "[InferenceRunnerFactoryTest]"),
            std::runtime_error);
    }

    TEST(Test__InferenceRunnerFactory_MoEOverlayPlanning, OverlayExecutionDevicePreservesGraphNativeContinuationParticipant)
    {
        GraphConfig graph_config;
        graph_config.moe.routed_expert_plan = makeActiveRocmLocalTPReplicatedOverlayPlan();

        const DeviceId effective_device = resolveMoEExpertOverlayExecutionDeviceForGraph(
            graph_config,
            nullptr,
            DeviceId::rocm(1),
            "[InferenceRunnerFactoryTest]");

        EXPECT_EQ(effective_device, DeviceId::rocm(1));
        ASSERT_NE(graph_config.moe.expert_overlay_runtime_plan, nullptr);
        const auto &continuation_domain =
            graph_config.moe.expert_overlay_runtime_plan->continuationDomain();
        EXPECT_TRUE(continuation_domain.domain_scoped_collective_context_ready);
    }

    TEST(Test__InferenceRunnerFactory_MoEOverlayPlanning, RebalanceControllersIncludeRoutedOverlayDomains)
    {
        GraphConfig graph_config;
        graph_config.n_layers = kMoELayers;
        graph_config.moe.num_experts = kMoEExperts;
        graph_config.moe.top_k = 2;
        graph_config.moe.rebalance_config.mode = MoERebalanceRuntimeMode::Observe;
        graph_config.moe.rebalance_config.window_size = 32;
        graph_config.moe.hot_expert_cache.kind = MoEHotExpertCacheConfig::Kind::Count;
        graph_config.moe.hot_expert_cache.count = 2;
        graph_config.moe.expert_overlay_runtime_plan =
            resolveMoEExpertOverlayRuntimePlan(makeRequestedOverlayPlan());

        auto controllers = createMoERebalanceControllersForGraph(
            graph_config,
            nullptr,
            nullptr);

        ASSERT_EQ(controllers.size(), 3u);
        EXPECT_EQ(controllers[0]->domainId(), "single");
        EXPECT_EQ(controllers[1]->domainId(), "overlay_routed_gpu_hot");
        EXPECT_EQ(controllers[2]->domainId(), "overlay_routed_cpu_cold");
        EXPECT_EQ(controllers[1]->participantCount(), 1);
        EXPECT_EQ(controllers[2]->participantCount(), 1);
        EXPECT_EQ(controllers[1]->mode(), MoERebalanceMode::OBSERVE);
        EXPECT_EQ(controllers[2]->mode(), MoERebalanceMode::OBSERVE);
        EXPECT_EQ(controllers[0]->maxReplicasPerSocket(), 2);
        EXPECT_EQ(controllers[1]->maxReplicasPerSocket(), 2);
        EXPECT_EQ(controllers[2]->maxReplicasPerSocket(), 2);
    }

    TEST(Test__InferenceRunnerFactory_MoEOverlayPlanning, GraphHistogramBindsToActiveOverlayRebalanceController)
    {
        GraphConfig graph_config;
        graph_config.n_layers = kMoELayers;
        graph_config.moe.num_experts = kMoEExperts;
        graph_config.moe.top_k = 2;
        graph_config.moe.rebalance_config.mode = MoERebalanceRuntimeMode::Dynamic;
        graph_config.moe.rebalance_config.window_size = 32;
        graph_config.moe.hot_expert_cache.kind = MoEHotExpertCacheConfig::Kind::Count;
        graph_config.moe.hot_expert_cache.count = 2;
        graph_config.moe.expert_overlay_runtime_plan =
            resolveMoEExpertOverlayRuntimePlan(makeRequestedOverlayPlan());

        auto controllers = createMoERebalanceControllersForGraph(
            graph_config,
            nullptr,
            nullptr);

        ASSERT_EQ(controllers.size(), 3u);
        ASSERT_EQ(controllers.front()->domainId(), "single");

        auto *active = bindActiveMoERebalanceControllerForGraph(
            graph_config,
            controllers);

        ASSERT_NE(active, nullptr);
        EXPECT_EQ(active->domainId(), "overlay_routed_gpu_hot");
        ASSERT_NE(active->histogram(), nullptr);
        EXPECT_EQ(graph_config.moe.decode_histogram, active->histogram());
        EXPECT_NE(graph_config.moe.decode_histogram, controllers.front()->histogram());
        EXPECT_EQ(graph_config.moe.rebalance_mode, active->mode());
    }

    TEST(Test__InferenceRunnerFactory_MoEOverlayPlanning, LLEPModeUsesDynamicControllerClock)
    {
        GraphConfig graph_config;
        graph_config.n_layers = kMoELayers;
        graph_config.moe.num_experts = kMoEExperts;
        graph_config.moe.top_k = 2;
        graph_config.moe.rebalance_config.mode = MoERebalanceRuntimeMode::LLEP;
        graph_config.moe.rebalance_config.window_size = 32;
        graph_config.moe.hot_expert_cache.kind = MoEHotExpertCacheConfig::Kind::Off;
        graph_config.moe.expert_overlay_runtime_plan =
            resolveMoEExpertOverlayRuntimePlan(makeActiveCudaLocalTPReplicatedOverlayPlan());

        auto controllers = createMoERebalanceControllersForGraph(
            graph_config,
            nullptr,
            nullptr);

        ASSERT_EQ(controllers.size(), 2u);
        ASSERT_EQ(controllers.front()->domainId(), "single");

        auto *active = bindActiveMoERebalanceControllerForGraph(
            graph_config,
            controllers);

        ASSERT_NE(active, nullptr);
        EXPECT_EQ(active->domainId(), "overlay_routed_cuda_hot");
        EXPECT_EQ(active->mode(), MoERebalanceMode::DYNAMIC)
            << "LLEP is a first-class public strategy, but it still uses the graph-captured dynamic maintenance clock";
        EXPECT_EQ(active->maxReplicasPerSocket(), 0)
            << "LLEP without an explicit hot-cache layer must not implicitly enable replicas";
        EXPECT_EQ(graph_config.moe.decode_histogram, active->histogram());
        EXPECT_EQ(graph_config.moe.rebalance_mode, MoERebalanceMode::DYNAMIC);
    }

    TEST(Test__InferenceRunnerFactory_MoEOverlayPlanning, HomogeneousGpuOwnershipMovesAreBoundedByLayerFanout)
    {
        GraphConfig graph_config;
        graph_config.n_layers = 40;
        graph_config.moe.num_experts = 256;
        graph_config.moe.top_k = 8;
        graph_config.moe.rebalance_config.mode = MoERebalanceRuntimeMode::Dynamic;
        graph_config.moe.rebalance_config.window_size = 256;
        graph_config.moe.expert_overlay_runtime_plan =
            resolveMoEExpertOverlayRuntimePlan(makeActiveCudaLocalTPReplicatedOverlayPlan());

        auto controllers = createMoERebalanceControllersForGraph(
            graph_config,
            nullptr,
            nullptr);

        auto it = std::find_if(
            controllers.begin(),
            controllers.end(),
            [](const std::unique_ptr<MoERebalanceController> &controller)
            {
                return controller && controller->domainId() == "overlay_routed_cuda_hot";
            });
        ASSERT_NE(it, controllers.end());

        const SocketRebalanceConfig &policy = (*it)->rebalanceConfig();
        EXPECT_EQ(policy.max_total_swaps, 6)
            << "Qwen-sized GPU ownership moves must fit the first-class staging arena";
        EXPECT_EQ(policy.max_swaps_per_layer, 3);
    }

    TEST(Test__InferenceRunnerFactory_MoEOverlayPlanning, GlobalTPRebalanceControllerPreservesCpuDomain)
    {
        GraphConfig graph_config;
        graph_config.n_layers = kMoELayers;
        graph_config.moe.num_experts = kMoEExperts;
        graph_config.moe.top_k = 2;
        graph_config.moe.rebalance_config.mode = MoERebalanceRuntimeMode::Dynamic;
        graph_config.moe.rebalance_config.window_size = 32;

        FakeGlobalTPContext global_tp(/*domain_id=*/17, /*my_index=*/1, /*degree=*/2);
        auto controllers = createMoERebalanceControllersForGraph(
            graph_config,
            nullptr,
            &global_tp);

        ASSERT_EQ(controllers.size(), 1u);
        const auto &controller = *controllers.front();
        EXPECT_EQ(controller.domainId(), "global_tp_domain_17");
        EXPECT_EQ(controller.participantCount(), 2);
        ASSERT_EQ(controller.participantDevices().size(), 2u);
        EXPECT_EQ(controller.participantDevices()[0], DeviceId(DeviceType::CPU, 0));
        EXPECT_EQ(controller.participantDevices()[1], DeviceId(DeviceType::CPU, 1));
        EXPECT_EQ(controller.mode(), MoERebalanceMode::DYNAMIC);
        EXPECT_EQ(controller.rebalanceConfig().max_total_swaps, 16)
            << "CPU domains keep the historical socket-rebalancer default";

        auto rank0_masks = controller.computeExpertMasksForParticipant(0);
        auto rank1_masks = controller.computeExpertMasksForParticipant(1);
        ASSERT_EQ(rank0_masks.size(), static_cast<size_t>(kMoELayers));
        ASSERT_EQ(rank1_masks.size(), static_cast<size_t>(kMoELayers));
        EXPECT_TRUE(rank0_masks[0][0]);
        EXPECT_TRUE(rank0_masks[0][2]);
        EXPECT_FALSE(rank0_masks[0][3]);
        EXPECT_FALSE(rank1_masks[0][2]);
        EXPECT_TRUE(rank1_masks[0][3]);
        EXPECT_TRUE(rank1_masks[0][5]);
    }

    // =============================================================================
    // Config Validation Tests
    // =============================================================================

    TEST(Test__RankOrchestratorConfig, ValidateEmptyDevicesFails)
    {
        RankOrchestrator::Config config;
        EXPECT_FALSE(config.validate());
    }

    TEST(Test__RankOrchestratorConfig, ValidateSingleDeviceSucceeds)
    {
        RankOrchestrator::Config config;
        config.devices = {GlobalDeviceAddress::cpu()};
        EXPECT_TRUE(config.validate());
    }

    TEST(Test__RankOrchestratorConfig, ValidateTwoDevicesSucceeds)
    {
        RankOrchestrator::Config config;
        config.devices = {GlobalDeviceAddress::cpu(), GlobalDeviceAddress::cpu()};
        EXPECT_TRUE(config.validate());
    }

    TEST(Test__RankOrchestratorConfig, ValidateWithEqualWeightsSucceeds)
    {
        RankOrchestrator::Config config;
        config.devices = {GlobalDeviceAddress::cpu(), GlobalDeviceAddress::cpu()};
        config.weights = {0.5f, 0.5f};
        EXPECT_TRUE(config.validate());
    }

    TEST(Test__RankOrchestratorConfig, ValidateWithUnequalWeightsSucceeds)
    {
        RankOrchestrator::Config config;
        config.devices = {GlobalDeviceAddress::cpu(), GlobalDeviceAddress::cpu()};
        config.weights = {0.73f, 0.27f};
        EXPECT_TRUE(config.validate());
    }

    TEST(Test__RankOrchestratorConfig, ValidateWeightCountMismatchFails)
    {
        RankOrchestrator::Config config;
        config.devices = {GlobalDeviceAddress::cpu(), GlobalDeviceAddress::cpu()};
        config.weights = {1.0f}; // Only one weight for two devices
        EXPECT_FALSE(config.validate());
    }

    TEST(Test__RankOrchestratorConfig, ValidateWeightsSumWrongFails)
    {
        RankOrchestrator::Config config;
        config.devices = {GlobalDeviceAddress::cpu(), GlobalDeviceAddress::cpu()};
        config.weights = {0.4f, 0.4f}; // Sum to 0.8, not 1.0
        EXPECT_FALSE(config.validate());
    }

    TEST(Test__RankOrchestratorConfig, GetNormalizedWeightsDefault)
    {
        RankOrchestrator::Config config;
        config.devices = {GlobalDeviceAddress::cpu(), GlobalDeviceAddress::cpu()};
        // No weights set

        auto weights = config.getNormalizedWeights();
        ASSERT_EQ(weights.size(), 2u);
        EXPECT_FLOAT_EQ(weights[0], 0.5f);
        EXPECT_FLOAT_EQ(weights[1], 0.5f);
    }

    TEST(Test__RankOrchestratorConfig, GetNormalizedWeightsExplicit)
    {
        RankOrchestrator::Config config;
        config.devices = {GlobalDeviceAddress::cpu(), GlobalDeviceAddress::cpu()};
        config.weights = {0.7f, 0.3f};

        auto weights = config.getNormalizedWeights();
        ASSERT_EQ(weights.size(), 2u);
        EXPECT_FLOAT_EQ(weights[0], 0.7f);
        EXPECT_FLOAT_EQ(weights[1], 0.3f);
    }

    TEST(Test__RankOrchestratorConfig, GetNormalizedWeightsThreeDevices)
    {
        RankOrchestrator::Config config;
        config.devices = {
            GlobalDeviceAddress::cpu(),
            GlobalDeviceAddress::cpu(),
            GlobalDeviceAddress::cpu()};
        // No weights set - should get equal distribution

        auto weights = config.getNormalizedWeights();
        ASSERT_EQ(weights.size(), 3u);
        EXPECT_NEAR(weights[0], 1.0f / 3.0f, 0.0001f);
        EXPECT_NEAR(weights[1], 1.0f / 3.0f, 0.0001f);
        EXPECT_NEAR(weights[2], 1.0f / 3.0f, 0.0001f);
    }

    TEST(Test__RankOrchestratorConfig, ValidateWeightsWithSmallTolerance)
    {
        // Test that weights summing to ~1.0 with floating point error still validate
        RankOrchestrator::Config config;
        config.devices = {GlobalDeviceAddress::cpu(), GlobalDeviceAddress::cpu()};
        config.weights = {0.6666666f, 0.3333334f}; // Sum = 1.0000000

        EXPECT_TRUE(config.validate());
    }

    TEST(Test__RankOrchestratorConfig, ValidateMixedDeviceTypes)
    {
        // Test heterogeneous device configuration
        RankOrchestrator::Config config;
        config.devices = {GlobalDeviceAddress::cuda(0), GlobalDeviceAddress::rocm(0)};
        config.weights = {0.73f, 0.27f};

        EXPECT_TRUE(config.validate());
    }

    TEST(Test__RankOrchestratorConfig, DefaultBackendIsAuto)
    {
        RankOrchestrator::Config config;
        EXPECT_EQ(config.backend, CollectiveBackendType::AUTO);
    }

    TEST(Test__RankOrchestratorConfig, DefaultMaxSeqLen)
    {
        RankOrchestrator::Config config;
        EXPECT_EQ(config.max_seq_len, 4096u);
    }

    TEST(Test__RankOrchestratorConfig, DefaultBatchSize)
    {
        RankOrchestrator::Config config;
        EXPECT_EQ(config.batch_size, 1);
    }

} // namespace
