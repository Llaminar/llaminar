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
#include "execution/moe/MoEOverlayParticipantResidency.h"
#include "execution/moe/MoEOverlayResidencyAuthority.h"
#include "execution/moe/MoEOverlayInferenceTransaction.h"
#include "execution/moe/MoERebalanceController.h"
#include "execution/moe/MoEExpertOverlayRuntimePlan.h"
#include "execution/moe/MoERoutedExpertPlacementPlanner.h"
#include "collective/ILocalTPContext.h"
#include "collective/IGlobalTPContext.h"
#include "backends/GlobalDeviceAddress.h"
#include "models/GraphTypes.h"
#include "mocks/MockModelContext.h"

#include <fstream>
#include <filesystem>
#include <sstream>
#include <string>
#include <string_view>

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
        namespace fs = std::filesystem;

        const fs::path relative_path(path);
        std::vector<fs::path> search_roots;
        search_roots.push_back(fs::current_path());

        /*
         * CTest normally launches this binary from the repository root, while
         * developers commonly invoke it directly from the build tree. Anchor a
         * second search at this translation unit so source-policy coverage is
         * independent of the caller's working directory.
         */
        fs::path source_anchor = fs::path(__FILE__).parent_path();
        if (source_anchor.is_relative())
            source_anchor = fs::absolute(source_anchor);
        search_roots.push_back(std::move(source_anchor));

        fs::path resolved_path;
        for (fs::path root : search_roots)
        {
            while (!root.empty())
            {
                const fs::path candidate = root / relative_path;
                if (fs::is_regular_file(candidate))
                {
                    resolved_path = candidate;
                    break;
                }

                const fs::path parent = root.parent_path();
                if (parent == root)
                    break;
                root = parent;
            }
            if (!resolved_path.empty())
                break;
        }

        if (resolved_path.empty())
            return {};

        std::ifstream input(resolved_path);
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

    /**
     * @brief Remove formatting whitespace from a source-contract fragment.
     *
     * Source-contract tests should protect ownership and lifecycle expressions,
     * not the formatter's current indentation or line wrapping. The returned
     * text retains every non-whitespace token, allowing assertions to describe
     * the required C++ expression without coupling the test to clang-format.
     *
     * @param source Source fragment whose token adjacency should be inspected.
     * @return A copy containing no ASCII whitespace characters.
     */
    std::string withoutFactorySourceWhitespace(std::string_view source)
    {
        std::string compact;
        compact.reserve(source.size());
        for (const char ch : source)
        {
            if (ch != ' ' && ch != '\t' && ch != '\n' && ch != '\r' && ch != '\f' && ch != '\v')
                compact.push_back(ch);
        }
        return compact;
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
        bool reserveCollectiveResources(size_t /*bytes*/, size_t /*fp16_scratch_elements*/) override { return true; }

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
        bool gatherVariableFloatRecordsToRoot(
            const float *, size_t, float *, size_t, size_t, int,
            size_t &, const std::string &) override
        {
            return false;
        }
        bool broadcastFloatElements(
            TensorBase *, size_t, int, const std::string &) override
        {
            return false;
        }
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

    /**
     * @brief Build a tiny MoE model whose final raw block is a real NextN sidecar.
     *
     * Qwen3.6 reports four raw blocks in this reduced geometry, but only layers
     * zero through two belong to the ordinary decoder.  The complete layer-three
     * tensor inventory lets the production manifest resolver prove that its FFN
     * is routed MoE rather than inferring that fact from a test-only flag.
     */
    std::shared_ptr<MockModelContext> makeMoEModelContextWithTrailingMTP()
    {
        constexpr int kRawLayerCount = kMoELayers + 1;
        auto model_ctx = MockModelContextBuilder()
                             .setArchitecture("qwen35moe")
                             .setBlockCount(kRawLayerCount)
                             .setEmbeddingLength(kMoEDModel)
                             .setHeadCount(4)
                             .setHeadCountKV(2)
                             .setVocabSize(128)
                             .setContextLength(256)
                             .setFeedForwardLength(kMoEIntermediate)
                             .build();

        auto &loader = model_ctx->mockLoader();
        loader.setIntParam("qwen35moe.expert_count", kMoEExperts);
        loader.setIntParam(
            "qwen35moe.expert_feed_forward_length",
            kMoEIntermediate);
        loader.setIntParam("qwen35moe.expert_shared_count", 1);
        loader.setIntParam("qwen35moe.nextn_predict_layers", 1);

        const std::string prefix =
            "blk." + std::to_string(kMoELayers) + ".";
        for (const char *suffix : {
                 "nextn.eh_proj.weight",
                 "nextn.hnorm.weight",
                 "nextn.enorm.weight",
                 "nextn.shared_head_norm.weight",
                 "attn_norm.weight",
                 "attn_q.weight",
                 "attn_k.weight",
                 "attn_v.weight",
                 "attn_output.weight",
                 "attn_q_norm.weight",
                 "attn_k_norm.weight",
                 "post_attention_norm.weight",
                 "ffn_gate_inp.weight",
                 "ffn_gate_exps.weight",
                 "ffn_up_exps.weight",
                 "ffn_down_exps.weight",
                 "ffn_gate_shexp.weight",
                 "ffn_up_shexp.weight",
                 "ffn_down_shexp.weight",
                 "ffn_gate_inp_shexp.weight",
             })
        {
            loader.addFP32ZerosTensor(prefix + suffix, {4, 4});
        }
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
        rocm_hot.scope = ExecutionDomainScope::RANK_LOCAL;
        rocm_hot.backend = CollectiveBackendType::RCCL;
        rocm_hot.routed_compute_policy = RoutedExpertComputePolicy::TensorSharded;
        /* Both local GPU participants belong to the current MPI rank. */
        rocm_hot.owner_rank = 0;
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
        cuda_hot.scope = ExecutionDomainScope::RANK_LOCAL;
        cuda_hot.backend = CollectiveBackendType::NCCL;
        cuda_hot.routed_compute_policy = RoutedExpertComputePolicy::Apportioned;
        /* Both local GPU participants belong to the current MPI rank. */
        cuda_hot.owner_rank = 0;
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
     * @brief Factory forwarding preserves the complete model reuse contract.
     *
     * The unit binary deliberately does not initialize MPI, while the concrete
     * factory consults the production rank context. Guard the more fundamental
     * source invariant instead: one optional typed contract crosses the private
     * boundary and is moved directly into OrchestrationRunner. This makes all
     * present and future ownership fields indivisible at compile time.
     */
    TEST(Test__InferenceRunnerFactory_SourceContract,
         PreparedModelReuseContractRemainsIndivisible)
    {
        const std::string source = readFactorySourceFile(
            "src/v2/execution/runner/OrchestrationRunnerFactory.cpp");
        ASSERT_FALSE(source.empty());
        const std::string compact = withoutFactorySourceWhitespace(source);

        EXPECT_NE(
            compact.find(
                "std::optional<ModelContextReuseContract>reuse_contract"),
            std::string::npos)
            << "the private construction boundary must retain the typed contract";
        EXPECT_NE(
            compact.find(
                "createFromOrchestrationConfigImpl(std::move(config),nullptr,std::move(reuse_contract))"),
            std::string::npos)
            << "the public overload must forward the complete contract";
        EXPECT_NE(
            compact.find("std::move(*reuse_contract)"),
            std::string::npos)
            << "OrchestrationRunner must consume the original contract object";
        EXPECT_EQ(
            compact.find(
                "automodel_context=std::move(reuse_contract.context)"),
            std::string::npos)
            << "member-wise decomposition can silently drop new ownership fields";
    }

    /**
     * @brief Every production device-graph factory receives workspace authority.
     *
     * A prepared model can execute through the ordinary rank-local graph,
     * unified pipeline graph, explicit PP stage, or interface-testable graph.
     * Missing the authority in even one construction path makes that runner
     * free model-lifetime backing and causes the next MTP family to allocate a
     * second workspace. Guard all four typed dependency sites together.
     */
    TEST(Test__InferenceRunnerFactory_SourceContract,
         ProductionGraphFactoriesPropagateReusableWorkspaceAuthority)
    {
        const std::string source =
            readFactorySourceFile(
                "src/v2/execution/factory/InferenceRunnerFactory.cpp");
        ASSERT_FALSE(source.empty());
        const std::string compact = withoutFactorySourceWhitespace(source);

        EXPECT_GE(
            countFactorySourceOccurrences(
                compact,
                "deps.reusable_execution_workspaces=config.reusable_execution_workspaces;"),
            4u)
            << "every device-graph construction path must share model-lifetime workspace ownership";
        EXPECT_NE(
            compact.find("deps.mpi_ctx=mpi_ctx;"),
            std::string::npos)
            << "the ordinary production graph must retain its concrete MPI authority in typed dependencies";
        EXPECT_EQ(
            compact.find(
                "std::make_unique<DeviceGraphOrchestrator>(std::move(graph_builder),mpi_ctx)"),
            std::string::npos)
            << "the legacy constructor cannot carry prepared-model workspace ownership";
    }

    /**
     * @brief Primary and mirrored-head plans must share one prepared store.
     *
     * NodeTP materializes the ordinary sharded model before its replicated
     * MTP terminal head. Creating a new PreparedWeightStore for the second plan
     * loses every primary prepared ref after CPU packing has released raw tensor
     * bytes. The factory must compose both plans additively in the existing
     * model-owned store and reject a competing explicit store.
     */
    TEST(Test__InferenceRunnerFactory_SourceContract,
         MirroredMTPHeadPlanReusesModelOwnedPreparedStore)
    {
        const std::string source =
            readFactorySourceFile(
                "src/v2/execution/factory/InferenceRunnerFactory.cpp");
        ASSERT_FALSE(source.empty());

        const size_t helper_begin = source.find(
            "installPreparedWeightStoreForPlan(");
        ASSERT_NE(helper_begin, std::string::npos);
        const size_t helper_end = source.find(
            "\n    namespace\n    {",
            helper_begin);
        ASSERT_NE(helper_end, std::string::npos);
        const std::string helper =
            source.substr(helper_begin, helper_end - helper_begin);
        const std::string compact_helper =
            withoutFactorySourceWhitespace(helper);

        EXPECT_NE(
            compact_helper.find("weight_mgr.preparedWeightStoreIfInitialized()"),
            std::string::npos)
            << "successive primary/mirrored plans must discover the existing model store";
        EXPECT_NE(
            compact_helper.find("installed_store!=config.prepared_weight_store"),
            std::string::npos)
            << "a second explicit prepared authority must fail closed";
        EXPECT_NE(
            compact_helper.find(":installed_store?installed_store:"),
            std::string::npos)
            << "the implicit path must reuse, not replace, the installed store";
        EXPECT_NE(
            compact_helper.find("if(!installed_store)weight_mgr.setPreparedWeightStore(store)"),
            std::string::npos)
            << "store installation must happen exactly once per model authority";
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

    /**
     * @brief Replicated routed compute must materialize complete expert tensors.
     *
     * Graph lowering cannot make an expert device-local when the preceding
     * LocalTP weight plan has already sliced that expert away. Keep the
     * resolved graph policy wired into every TP-aware primary weight-plan site,
     * while the plan builder applies the bypass only to routed-expert roles.
     */
    TEST(Test__InferenceRunnerFactory_SourceContract,
         ReplicatedRoutedComputeBypassesExpertAxisSlicing)
    {
        const std::string source =
            readFactorySourceFile(
                "src/v2/execution/factory/InferenceRunnerFactory.cpp");
        ASSERT_FALSE(source.empty());

        EXPECT_NE(
            source.find(
                "graph_config.moe.routed_compute_policy ==\n"
                "                   RoutedExpertComputePolicy::Replicated"),
            std::string::npos)
            << "weight materialization must derive full expert residency from the typed graph policy";
        EXPECT_NE(
            source.find(
                "isRoutedExpertRole(inferWeightRole(weight_name))"),
            std::string::npos)
            << "the TP bypass must be limited to routed-expert tensors";
        EXPECT_NE(
            source.find(
                "options.bypass_tensor_parallel || replicate_routed_weight"),
            std::string::npos)
            << "replicated routed requirements must select the full source tensor";
        EXPECT_GE(
            countFactorySourceOccurrences(
                source,
                "needsReplicatedRoutedExpertWeights(graph_config)"),
            3u)
            << "single, nested TP-in-PP, and LocalTP primary plans must share the same residency contract";
    }

    /**
     * @brief Every LocalTP graph must freeze only its exact overlay slices.
     *
     * Device runners on one MPI rank are built concurrently and share an
     * additive PreparedWeightStore.  Their FrozenModelWeightSets remain
     * graph-local: otherwise the CUDA:0 graph can page, prepare, or publish the
     * CUDA:1 participant's experts and graph construction later observes an
     * incomplete registry on the actual owner.  Guard both the exact
     * rank-and-device filter and its use at the ordinary and LocalTP materialize
     * sites without requiring a real GGUF in this fast unit suite.
     */
    TEST(Test__InferenceRunnerFactory_SourceContract,
         ExpertOverlayWeightPlansAreGraphLocalInLocalTP)
    {
        const std::string source =
            readFactorySourceFile(
                "src/v2/execution/factory/InferenceRunnerFactory.cpp");
        ASSERT_FALSE(source.empty());

        EXPECT_NE(source.find("participant.world_rank == rank &&"),
                  std::string::npos)
            << "overlay slice selection must authenticate the exact MPI rank";
        EXPECT_NE(source.find("participant.device == graph_device ||"),
                  std::string::npos)
            << "ordinary LocalTP slices must authenticate the exact graph device";
        EXPECT_NE(source.find("graph_owns_colocated_cpu_endpoints &&"),
                  std::string::npos)
            << "only the continuation-root graph may add colocated CPU sparse endpoints";
        EXPECT_GE(
            countFactorySourceOccurrences(
                source,
                "includeGraphLocalOverlayParticipantWeights("),
            3u)
            << "the helper definition plus concrete and LocalTP materialization sites must remain wired";
        EXPECT_EQ(
            source.find("rankLocalOverlayPreparationDevices"),
            std::string::npos)
            << "one device graph must never prepare every sibling device on its MPI rank";
    }

    /**
     * @brief LLEP selection must validate, never synthesize, graph policy.
     *
     * The routed domain declaration is the authoritative description of
     * compute residency, phase scheduling, and row assignment. Treating the
     * maintenance-mode selector as an implicit assignment override makes the
     * graph depend on factory control flow and previously hid a decode
     * collective behind an apportioned outer policy.
     */
    TEST(Test__InferenceRunnerFactory_SourceContract,
         CurrentBatchLLEPPolicyIsValidatedAndNeverSynthesized)
    {
        const std::string source =
            readFactorySourceFile(
                "src/v2/execution/factory/InferenceRunnerFactory.cpp");
        ASSERT_FALSE(source.empty());

        EXPECT_EQ(
            source.find("setRoutedOverlayLeastLoadedResidentAssignment"),
            std::string::npos)
            << "the runner factory must not mutate declarative routed policy";
        EXPECT_NE(
            source.find("validateCurrentBatchLLEPOverlayPolicy"),
            std::string::npos)
            << "current-batch LLEP must validate its declared graph policy before lowering";
        EXPECT_NE(
            source.find(
                "routed_prefill_assignment=least-loaded-resident"),
            std::string::npos)
            << "a missing explicit LLEP prefill assignment must fail with actionable diagnostics";
        EXPECT_NE(
            source.find("routed_decode_assignment=static-owner"),
            std::string::npos)
            << "the economical grouped-verifier assignment must be validated independently";
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
        EXPECT_TRUE(requested_plan->continuation_domain_spec.domain.empty());
        EXPECT_EQ(
            resolved_plan->continuation_domain_spec.domain,
            requested_plan->continuation_domain)
            << "model freezing must make the inherited continuation topology explicit";
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
        EXPECT_EQ(
            resolved_plan->authority_execution,
            MoEOverlayAuthorityExecutionKind::HostResident);
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

        ASSERT_NE(resolved_plan, nullptr);
        EXPECT_NE(resolved_plan, explicit_plan)
            << "model freezing must seal authority execution without mutating declarative input";
        EXPECT_EQ(
            explicit_plan->authority_execution,
            MoEOverlayAuthorityExecutionKind::Unresolved);
        EXPECT_EQ(
            resolved_plan->authority_execution,
            MoEOverlayAuthorityExecutionKind::HostResident);
        EXPECT_TRUE(explicit_plan->continuation_domain_spec.domain.empty());
        EXPECT_EQ(
            resolved_plan->continuation_domain_spec.domain,
            explicit_plan->continuation_domain)
            << "explicit placements must not bypass frozen topology completion";
        ASSERT_EQ(resolved_plan->placements.size(), 3u);
        EXPECT_EQ(resolved_plan->placements[0].routed_expert_tier,
                  (std::vector<int>{0, 1, 0, 1, 0, 1}));
        EXPECT_EQ(resolved_plan->placements[1].routed_expert_tier,
                  (std::vector<int>{1, 0, 1, 0, 1, 0}));
        EXPECT_EQ(resolved_plan->placements[2].routed_expert_tier,
                  (std::vector<int>{0, 0, 1, 1, 0, 1}));
    }

    /**
     * @brief Runtime MTP policy selects whether the routed NextN bank exists.
     *
     * This is the focused regression for the real Qwen3.6 ExpertOverlay
     * campaign: its serial graph built only forty main layers while residency
     * incorrectly waited for layer forty, and the MTP graph needs that same
     * layer forty bank. Both plans must be derived from one immutable request.
     */
    TEST(Test__InferenceRunnerFactory_MoEOverlayPlanning,
         RuntimeMTPPolicySelectsTrailingRoutedSidecarLayer)
    {
        auto model_ctx = makeMoEModelContextWithTrailingMTP();
        auto requested_plan = makeRequestedOverlayPlan();

        InferenceRunnerConfig serial_config;
        serial_config.moe_routed_expert_plan = requested_plan;
        const auto serial_plan =
            resolveMoERoutedExpertPlacementPlanForModel(
                *model_ctx,
                serial_config);

        InferenceRunnerConfig mtp_config = serial_config;
        mtp_config.mtp.enabled = true;
        mtp_config.mtp.draft_tokens = 3;
        const auto mtp_plan =
            resolveMoERoutedExpertPlacementPlanForModel(
                *model_ctx,
                mtp_config);

        InferenceRunnerConfig retained_control_config = serial_config;
        retained_control_config.mtp.enabled = false;
        retained_control_config.mtp.graph_capacity_draft_tokens = 15;
        const auto retained_control_plan =
            resolveMoERoutedExpertPlacementPlanForModel(
                *model_ctx,
                retained_control_config);
        const auto retained_control_metadata =
            resolveMoERoutedExpertModelMetadataForModel(
                *model_ctx,
                retained_control_config.mtp);

        ASSERT_NE(serial_plan, nullptr);
        ASSERT_NE(mtp_plan, nullptr);
        EXPECT_TRUE(requested_plan->placements.empty());
        ASSERT_EQ(serial_plan->placements.size(), 3u);
        EXPECT_EQ(serial_plan->placements.back().layer, 2);
        ASSERT_EQ(mtp_plan->placements.size(), 4u);
        EXPECT_EQ(mtp_plan->placements.back().layer, 3);
        ASSERT_NE(retained_control_plan, nullptr);
        ASSERT_EQ(retained_control_plan->placements.size(), 4u);
        EXPECT_EQ(retained_control_plan->placements.back().layer, 3);
        EXPECT_EQ(retained_control_metadata.num_layers, 4);
        EXPECT_EQ(
            retained_control_metadata.main_inference_layer_count,
            kMoELayers)
            << "retained MTP storage must not redefine the ordinary inference interval";
        EXPECT_EQ(
            retained_control_plan->lastPlacementLayerBefore(
                retained_control_metadata.main_inference_layer_count),
            std::optional<int>{kMoELayers - 1});

        const auto serial_family =
            resolveMoEOverlayInferenceGraphFamilyIdentity(
                *model_ctx->loader(),
                model_ctx->architecture(),
                model_ctx->totalBlockCount(),
                MoEOverlayMTPGraphFamilyPolicy::MainOnly,
                /*graph_family_generation=*/1,
                /*max_graph_rows=*/16,
                /*max_decode_rows=*/1,
                /*max_request_count=*/1,
                /*max_mtp_draft_depth=*/0);
        const auto mtp_family =
            resolveMoEOverlayInferenceGraphFamilyIdentity(
                *model_ctx->loader(),
                model_ctx->architecture(),
                model_ctx->totalBlockCount(),
                MoEOverlayMTPGraphFamilyPolicy::RetainModelSidecars,
                /*graph_family_generation=*/1,
                /*max_graph_rows=*/16,
                /*max_decode_rows=*/4,
                /*max_request_count=*/1,
                /*max_mtp_draft_depth=*/3);

        EXPECT_EQ(serial_family.routedLayerCapacity(), kMoELayers);
        EXPECT_EQ(
            serial_family.routedLayerCapacity(),
            serial_plan->placementLayerCapacity())
            << "an inactive raw NextN block must not enlarge follower runtime geometry";
        EXPECT_EQ(mtp_family.routedLayerCapacity(), kMoELayers + 1);
        EXPECT_EQ(
            mtp_family.routedLayerCapacity(),
            mtp_plan->placementLayerCapacity())
            << "an enabled routed NextN graph must share the placement bank's raw source slot";
    }

    /**
     * @brief An all-feature explicit plan produces an immutable serial view.
     */
    TEST(Test__InferenceRunnerFactory_MoEOverlayPlanning,
         NonMTPRuntimeClonesAndTrimsOnlyAuthenticatedTrailingSidecar)
    {
        auto model_ctx = makeMoEModelContextWithTrailingMTP();
        InferenceRunnerConfig mtp_config;
        mtp_config.moe_routed_expert_plan = makeRequestedOverlayPlan();
        mtp_config.mtp.enabled = true;
        auto all_feature_plan =
            resolveMoERoutedExpertPlacementPlanForModel(
                *model_ctx,
                mtp_config);
        ASSERT_NE(all_feature_plan, nullptr);
        ASSERT_EQ(all_feature_plan->placements.size(), 4u);

        InferenceRunnerConfig serial_config;
        serial_config.moe_routed_expert_plan = all_feature_plan;
        auto serial_plan = resolveMoERoutedExpertPlacementPlanForModel(
            *model_ctx,
            serial_config);

        ASSERT_NE(serial_plan, nullptr);
        EXPECT_NE(serial_plan, all_feature_plan);
        EXPECT_EQ(serial_plan->placements.size(), 3u);
        EXPECT_EQ(all_feature_plan->placements.size(), 4u)
            << "freezing a serial graph must not mutate the reusable declaration";
    }

    /**
     * @brief MTP cannot run against an explicit plan missing its sidecar bank.
     */
    TEST(Test__InferenceRunnerFactory_MoEOverlayPlanning,
         MTPRuntimeRejectsExplicitPlanWithoutRoutedSidecarLayer)
    {
        auto model_ctx = makeMoEModelContextWithTrailingMTP();
        InferenceRunnerConfig serial_config;
        serial_config.moe_routed_expert_plan = makeRequestedOverlayPlan();
        auto main_only_plan =
            resolveMoERoutedExpertPlacementPlanForModel(
                *model_ctx,
                serial_config);
        ASSERT_NE(main_only_plan, nullptr);
        ASSERT_EQ(main_only_plan->placements.size(), 3u);

        InferenceRunnerConfig mtp_config;
        mtp_config.moe_routed_expert_plan = main_only_plan;
        mtp_config.mtp.enabled = true;
        EXPECT_THROW(
            (void)resolveMoERoutedExpertPlacementPlanForModel(
                *model_ctx,
                mtp_config),
            std::invalid_argument);
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

    TEST(Test__InferenceRunnerFactory_MoEOverlayPlanning,
         OverlayRuntimePlanWithoutRCUAuthorityIsRejected)
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

        EXPECT_THROW(
            validateMoEDurableResidencyAuthorityForGraph(
                graph_config, nullptr, nullptr),
            std::logic_error);
    }

    TEST(Test__InferenceRunnerFactory_MoEOverlayPlanning,
         DynamicOverlayCannotBindLegacyController)
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

        EXPECT_THROW(
            validateMoEDurableResidencyAuthorityForGraph(
                graph_config, nullptr, nullptr),
            std::logic_error);
    }

    /**
     * @brief A live ExpertOverlay authority is the only representable writer.
     *
     * A one-domain overlay may have several participants, which is precisely
     * the geometry that made the legacy graph-side Dynamic lowering eligible.
     * `SingleDomain` must now bind the RCU owner map as the sole durable
     * authority before graph construction; graph configuration has no second
     * durable-writer mode to clear or reconcile.
     */
    TEST(Test__InferenceRunnerFactory_MoEOverlayPlanning,
         InstalledExpertOverlayRCUIsTheOnlyDurableResidencyAuthority)
    {
        auto model_ctx = makeMoEModelContext();
        InferenceRunnerConfig runner_config;
        auto single_domain_plan =
            makeActiveCudaLocalTPReplicatedOverlayPlan();
        single_domain_plan->topology =
            RoutedExpertPlacementTopology::SingleDomain;
        single_domain_plan->residency_policy =
            RoutedExpertResidencyPolicy::StaticById;
        runner_config.moe_routed_expert_plan =
            resolveMoERoutedExpertPlacementPlanForModel(
                *model_ctx,
                InferenceRunnerConfig{
                    .moe_routed_expert_plan =
                        std::move(single_domain_plan),
                });
        ASSERT_NE(runner_config.moe_routed_expert_plan, nullptr);

        auto authority = std::make_shared<MoEOverlayResidencyAuthority>(
            MoEOverlayResidencyAuthority::Config{
                .initial_plan = *runner_config.moe_routed_expert_plan,
                .model_metadata = {
                    .num_layers = kMoELayers,
                    .num_experts = kMoEExperts,
                    .d_model = kMoEDModel,
                    .routed_intermediate_size = kMoEIntermediate,
                },
                .maintenance_mode = MoERebalanceRuntimeMode::Off,
                .histogram = nullptr,
                .perf_device = "factory_unit",
            });
        const auto snapshot = authority->snapshot();
        ASSERT_NE(snapshot, nullptr);
        ASSERT_TRUE(snapshot->valid());

        std::vector<int> local_participant_ids;
        for (const auto &participant : snapshot->owner_map.participants())
            local_participant_ids.push_back(participant.participant_id);
        auto participant_residency =
            std::make_shared<MoEOverlayParticipantResidencyRegistry>(
                MoEOverlayParticipantResidencyRegistry::Config{
                    .owner_map = snapshot->owner_map,
                    .local_participant_ids = local_participant_ids,
                    .num_layers = kMoELayers,
                    .num_experts = kMoEExperts,
                    .initial_epoch = snapshot->epoch,
                });

        runner_config.moe_expert_overlay_residency_authority = authority;
        runner_config.moe_expert_overlay_participant_residency =
            participant_residency;
        runner_config.moe_rebalance.mode = MoERebalanceRuntimeMode::Off;

        GraphConfig graph_config;
        graph_config.moe.rebalance_config.mode =
            MoERebalanceRuntimeMode::Off;
        DomainTPContextMap owned_domain_tp_contexts;
        ASSERT_TRUE(applyMoEExpertOverlayConfigToGraphForTesting(
            *model_ctx,
            runner_config,
            nullptr,
            graph_config,
            owned_domain_tp_contexts));

        EXPECT_EQ(
            graph_config.moe.durable_residency_authority,
            MoEDurableResidencyAuthorityKind::ExpertOverlayRCU);
        EXPECT_EQ(
            graph_config.moe.expert_overlay_residency_authority,
            authority);

        EXPECT_NO_THROW(validateMoEDurableResidencyAuthorityForGraph(
            graph_config, nullptr, nullptr));
    }

    TEST(Test__InferenceRunnerFactory_MoEOverlayPlanning,
         CurrentBatchLLEPRequiresExpertOverlayParentAuthority)
    {
        GraphConfig graph_config;
        graph_config.n_layers = kMoELayers;
        graph_config.moe.num_experts = kMoEExperts;
        graph_config.moe.top_k = 2;
        graph_config.moe.routed_decode_assignment_policy =
            RoutedExpertAssignmentPolicy::StaticOwner;
        graph_config.moe.routed_prefill_assignment_policy =
            RoutedExpertAssignmentPolicy::LeastLoadedResident;
        graph_config.moe.rebalance_config.mode = MoERebalanceRuntimeMode::Off;
        graph_config.moe.rebalance_config.window_size = 32;
        graph_config.moe.hot_expert_cache.kind = MoEHotExpertCacheConfig::Kind::Off;
        graph_config.moe.expert_overlay_runtime_plan =
            resolveMoEExpertOverlayRuntimePlan(makeActiveCudaLocalTPReplicatedOverlayPlan());

        EXPECT_THROW(
            validateMoEDurableResidencyAuthorityForGraph(
                graph_config, nullptr, nullptr),
            std::logic_error)
            << "Current-batch LLEP must be a child of an installed durable "
               "ExpertOverlay epoch, never a reason to create another writer";
        EXPECT_EQ(
            graph_config.moe.durable_residency_authority,
            MoEDurableResidencyAuthorityKind::None);
    }

    TEST(Test__InferenceRunnerFactory_MoEOverlayPlanning,
         HomogeneousGpuDynamicCannotReenterLegacyController)
    {
        GraphConfig graph_config;
        graph_config.n_layers = 40;
        graph_config.moe.num_experts = 256;
        graph_config.moe.top_k = 8;
        graph_config.moe.rebalance_config.mode = MoERebalanceRuntimeMode::Dynamic;
        graph_config.moe.rebalance_config.window_size = 256;
        graph_config.moe.expert_overlay_runtime_plan =
            resolveMoEExpertOverlayRuntimePlan(makeActiveCudaLocalTPReplicatedOverlayPlan());

        EXPECT_THROW(
            validateMoEDurableResidencyAuthorityForGraph(
                graph_config, nullptr, nullptr),
            std::logic_error);
    }

    TEST(Test__InferenceRunnerFactory_MoEOverlayPlanning,
         GlobalTPRequiresNormalizedExpertOverlayAuthority)
    {
        GraphConfig graph_config;
        graph_config.n_layers = kMoELayers;
        graph_config.moe.num_experts = kMoEExperts;
        graph_config.moe.top_k = 2;
        graph_config.moe.rebalance_config.mode = MoERebalanceRuntimeMode::Dynamic;
        graph_config.moe.rebalance_config.window_size = 32;

        FakeGlobalTPContext global_tp(/*domain_id=*/17, /*my_index=*/1, /*degree=*/2);
        EXPECT_THROW(
            validateMoEDurableResidencyAuthorityForGraph(
                graph_config, nullptr, &global_tp),
            std::logic_error);
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
