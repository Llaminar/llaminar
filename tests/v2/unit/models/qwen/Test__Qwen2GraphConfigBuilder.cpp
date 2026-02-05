/**
 * @file Test__Qwen2GraphConfigBuilder.cpp
 * @brief Unit tests for Qwen2GraphConfigBuilder
 *
 * Tests:
 * - Building config for single device (no TP, no PP)
 * - Building config for PP (layer ranges, embedding/lm_head flags)
 * - Building config for LOCAL TP (head distribution)
 * - Building config for CPU spillover
 * - Head distribution with proportional weights
 *
 * @author David Sanftenberg
 * @date January 2026
 */

#include <gtest/gtest.h>
#include "models/qwen/Qwen2GraphConfigBuilder.h"
#include "models/qwen/Qwen2Graph.h" // For Qwen2GraphConfig
#include "models/IGraphConfigBuilder.h"
#include "execution/mpi_orchestration/RankExecutionPlan.h"
#include "config/OrchestrationConfig.h"
#include "config/TensorParallelConfig.h" // For setTensorParallelConfig stub
#include "backends/GlobalDeviceAddress.h"
#include "loaders/IWeightManager.h"

using namespace llaminar2;

// ============================================================================
// Minimal WeightManager stub for testing
// ============================================================================

namespace
{
    /**
     * @brief Minimal WeightManager stub for unit testing
     *
     * Qwen2GraphConfigBuilder doesn't actually use WeightManager in buildConfig,
     * so we provide a minimal stub that satisfies the interface.
     */
    class StubWeightManager : public IWeightManager
    {
    public:
        std::shared_ptr<TensorBase> getWeight(const std::string & /*name*/,
                                              DeviceId /*device*/, int /*layer_idx*/) override
        {
            return nullptr;
        }

        std::shared_ptr<TensorBase> getWeightForDevice(const std::string & /*name*/,
                                                       DeviceId /*device*/, int /*layer_idx*/) override
        {
            return nullptr;
        }

        bool preloadForDevices(const std::vector<DeviceId> & /*devices*/) override
        {
            return true;
        }

        std::shared_ptr<TensorBase> getDecodeWeight(const std::string & /*name*/,
                                                    DeviceId /*decode_device*/, float /*fraction*/, int /*layer_idx*/) override
        {
            return nullptr;
        }

        bool isWeightSharded(const std::string & /*name*/) const override
        {
            return false;
        }

        ShardingMode getShardingMode(const std::string & /*name*/) const override
        {
            return ShardingMode::REPLICATE;
        }

        bool isGemmWeight(const std::string & /*name*/) const override
        {
            return true;
        }

        WeightDistributionStrategy strategy() const override
        {
            return WeightDistributionStrategy::REPLICATED;
        }

        bool hasLMHead() const override { return true; }
        bool hasEmbedding() const override { return true; }

        size_t cacheSize() const override { return 0; }
        void clearCache() override {}
        size_t decodeCacheSize() const override { return 0; }
        void clearDecodeCache() override {}

        void setWeightShardingConfig(const WeightShardingConfig & /*config*/) override {}

        void setTensorParallelConfig(std::shared_ptr<TensorParallelConfig> /*config*/) override {}

        // New methods folded from WeightPreloader
        bool packGemmWeights(DeviceId /*target_device*/,
                             PreloadProgressCallback /*progress_cb*/,
                             bool /*release_raw_data*/) override { return true; }
        bool uploadNonGemmWeights(DeviceId /*target_device*/) override { return true; }
        std::pair<size_t, size_t> preloadStats() const override { return {0, 0}; }
    };

    // Global stub for tests to use
    StubWeightManager g_stub_weight_manager;
}

// ============================================================================
// Test Constants
// ============================================================================

namespace
{
    ModelConfig createTestModelConfig()
    {
        ModelConfig config;
        config.name = "TestQwen2";
        config.n_layers = 32;
        config.n_heads = 28;
        config.n_kv_heads = 4;
        config.hidden_size = 3584;        // d_model
        config.intermediate_size = 18944; // d_ff
        config.vocab_size = 151936;
        config.head_dim = 128;
        // Note: rms_norm_eps and rope_theta are not in ModelConfig;
        // they use defaults from Qwen2GraphConfig (1e-6f, 10000.0f)
        return config;
    }

    RankExecutionPlan createSingleDevicePlan()
    {
        RankExecutionPlan plan;
        plan.rank = 0;
        plan.hostname = "localhost";
        plan.numa_node = 0;
        plan.primary_device = GlobalDeviceAddress::cuda(0);
        plan.first_layer = 0;
        plan.last_layer = 31;
        plan.has_embedding = true;
        plan.has_lm_head = true;
        plan.tp_scope = TPScope::LOCAL;
        plan.weight_shard.shard_index = 0;
        plan.weight_shard.total_shards = 1;
        plan.weight_shard.work_fraction = 1.0f;
        return plan;
    }
} // namespace

// ============================================================================
// Factory Tests
// ============================================================================

TEST(Test__Qwen2GraphConfigBuilder, Factory_CreatesQwen2Builder)
{
    auto builder = createQwen2GraphConfigBuilder();
    ASSERT_NE(builder, nullptr);
}

TEST(Test__Qwen2GraphConfigBuilder, Factory_CreateByModelType)
{
    auto builder = createGraphConfigBuilder("qwen2");
    ASSERT_NE(builder, nullptr);

    builder = createGraphConfigBuilder("Qwen2");
    ASSERT_NE(builder, nullptr);

    builder = createGraphConfigBuilder("unsupported_model");
    EXPECT_EQ(builder, nullptr);
}

// ============================================================================
// Single Device Configuration Tests
// ============================================================================

class Test__Qwen2GraphConfigBuilder_SingleDevice : public ::testing::Test
{
protected:
    std::unique_ptr<IGraphConfigBuilder> builder;
    ModelConfig model_config;
    RankExecutionPlan plan;

    void SetUp() override
    {
        builder = createQwen2GraphConfigBuilder();
        model_config = createTestModelConfig();
        plan = createSingleDevicePlan();
    }
};

TEST_F(Test__Qwen2GraphConfigBuilder_SingleDevice, BuildConfig_ReturnsSuccess)
{
    // Use the generic buildConfig (doesn't need WeightManager for basic test)
    // We can't use buildQwen2Config without a real WeightManager
    auto result = builder->buildConfig(plan, model_config,
                                       g_stub_weight_manager);

    EXPECT_TRUE(result.success);
    EXPECT_TRUE(result.error.empty());
    EXPECT_NE(result.placement, nullptr);
}

TEST_F(Test__Qwen2GraphConfigBuilder_SingleDevice, BuildConfig_ExtractsModelInfo)
{
    auto result = builder->buildConfig(plan, model_config,
                                       g_stub_weight_manager);

    ASSERT_TRUE(result.success);

    EXPECT_EQ(result.model_info.n_layers, 32);
    EXPECT_EQ(result.model_info.n_heads, 28);
    EXPECT_EQ(result.model_info.n_kv_heads, 4);
    EXPECT_EQ(result.model_info.d_model, 3584);
    EXPECT_EQ(result.model_info.d_ff, 18944);
    EXPECT_EQ(result.model_info.vocab_size, 151936);
}

TEST_F(Test__Qwen2GraphConfigBuilder_SingleDevice, BuildConfig_NoSharding)
{
    auto result = builder->buildConfig(plan, model_config,
                                       g_stub_weight_manager);

    ASSERT_TRUE(result.success);

    EXPECT_EQ(result.execution_info.shard_index, 0);
    EXPECT_EQ(result.execution_info.total_shards, 1);
    EXPECT_EQ(result.execution_info.local_heads, model_config.n_heads);
    EXPECT_EQ(result.execution_info.local_kv_heads, model_config.n_kv_heads);
}

TEST_F(Test__Qwen2GraphConfigBuilder_SingleDevice, BuildConfig_AllLayers)
{
    auto result = builder->buildConfig(plan, model_config,
                                       g_stub_weight_manager);

    ASSERT_TRUE(result.success);

    EXPECT_EQ(result.execution_info.first_layer, 0);
    EXPECT_EQ(result.execution_info.last_layer, 31);
    EXPECT_TRUE(result.execution_info.has_embedding);
    EXPECT_TRUE(result.execution_info.has_lm_head);
}

// ============================================================================
// Pipeline Parallelism Tests
// ============================================================================

class Test__Qwen2GraphConfigBuilder_PP : public ::testing::Test
{
protected:
    std::unique_ptr<IGraphConfigBuilder> builder;
    ModelConfig model_config;

    void SetUp() override
    {
        builder = createQwen2GraphConfigBuilder();
        model_config = createTestModelConfig();
    }

    RankExecutionPlan createFirstStagePlan()
    {
        RankExecutionPlan plan;
        plan.rank = 0;
        plan.hostname = "node0";
        plan.numa_node = 0;
        plan.primary_device = GlobalDeviceAddress::cuda(0);
        plan.first_layer = 0;
        plan.last_layer = 15;
        plan.has_embedding = true;
        plan.has_lm_head = false;
        plan.next_rank = 1; // PP enabled
        plan.pp_stage_id = 0;
        plan.weight_shard.shard_index = 0;
        plan.weight_shard.total_shards = 1;
        plan.weight_shard.work_fraction = 1.0f;
        return plan;
    }

    RankExecutionPlan createLastStagePlan()
    {
        RankExecutionPlan plan;
        plan.rank = 1;
        plan.hostname = "node1";
        plan.numa_node = 0;
        plan.primary_device = GlobalDeviceAddress::cuda(0);
        plan.first_layer = 16;
        plan.last_layer = 31;
        plan.has_embedding = false;
        plan.has_lm_head = true;
        plan.prev_rank = 0; // PP enabled
        plan.pp_stage_id = 1;
        plan.weight_shard.shard_index = 0;
        plan.weight_shard.total_shards = 1;
        plan.weight_shard.work_fraction = 1.0f;
        return plan;
    }
};

TEST_F(Test__Qwen2GraphConfigBuilder_PP, FirstStage_LayerRange)
{
    auto plan = createFirstStagePlan();
    auto result = builder->buildConfig(plan, model_config,
                                       g_stub_weight_manager);

    ASSERT_TRUE(result.success);
    EXPECT_EQ(result.execution_info.first_layer, 0);
    EXPECT_EQ(result.execution_info.last_layer, 15);
}

TEST_F(Test__Qwen2GraphConfigBuilder_PP, FirstStage_HasEmbedding)
{
    auto plan = createFirstStagePlan();
    auto result = builder->buildConfig(plan, model_config,
                                       g_stub_weight_manager);

    ASSERT_TRUE(result.success);
    EXPECT_TRUE(result.execution_info.has_embedding);
    EXPECT_FALSE(result.execution_info.has_lm_head);
}

TEST_F(Test__Qwen2GraphConfigBuilder_PP, LastStage_LayerRange)
{
    auto plan = createLastStagePlan();
    auto result = builder->buildConfig(plan, model_config,
                                       g_stub_weight_manager);

    ASSERT_TRUE(result.success);
    EXPECT_EQ(result.execution_info.first_layer, 16);
    EXPECT_EQ(result.execution_info.last_layer, 31);
}

TEST_F(Test__Qwen2GraphConfigBuilder_PP, LastStage_HasLMHead)
{
    auto plan = createLastStagePlan();
    auto result = builder->buildConfig(plan, model_config,
                                       g_stub_weight_manager);

    ASSERT_TRUE(result.success);
    EXPECT_FALSE(result.execution_info.has_embedding);
    EXPECT_TRUE(result.execution_info.has_lm_head);
}

TEST_F(Test__Qwen2GraphConfigBuilder_PP, PlacementRespectsLayerRange)
{
    auto plan = createFirstStagePlan();
    auto result = builder->buildConfig(plan, model_config,
                                       g_stub_weight_manager);

    ASSERT_TRUE(result.success);
    ASSERT_NE(result.placement, nullptr);

    // First stage should build layers 0-15
    for (int i = 0; i <= 15; ++i)
    {
        EXPECT_TRUE(result.placement->shouldBuildLayer(i))
            << "Should build layer " << i;
    }
    for (int i = 16; i < 32; ++i)
    {
        EXPECT_FALSE(result.placement->shouldBuildLayer(i))
            << "Should not build layer " << i;
    }
}

// ============================================================================
// Local Tensor Parallelism Tests
// ============================================================================

class Test__Qwen2GraphConfigBuilder_LocalTP : public ::testing::Test
{
protected:
    std::unique_ptr<IGraphConfigBuilder> builder;
    ModelConfig model_config;

    void SetUp() override
    {
        builder = createQwen2GraphConfigBuilder();
        model_config = createTestModelConfig();
    }

    RankExecutionPlan createTwoDeviceTPPlan()
    {
        RankExecutionPlan plan;
        plan.rank = 0;
        plan.hostname = "localhost";
        plan.numa_node = 0;
        plan.primary_device = GlobalDeviceAddress::cuda(0);
        plan.first_layer = 0;
        plan.last_layer = 31;
        plan.has_embedding = true;
        plan.has_lm_head = true;
        plan.tp_scope = TPScope::LOCAL;
        plan.local_tp_devices = {
            GlobalDeviceAddress::cuda(0),
            GlobalDeviceAddress::cuda(1)};
        plan.local_tp_weights = {0.5f, 0.5f};
        plan.weight_shard.shard_index = 0;
        plan.weight_shard.total_shards = 2;
        plan.weight_shard.work_fraction = 0.5f;
        return plan;
    }

    RankExecutionPlan createProportionalTPPlan()
    {
        RankExecutionPlan plan;
        plan.rank = 0;
        plan.hostname = "localhost";
        plan.numa_node = 0;
        plan.primary_device = GlobalDeviceAddress::cuda(0);
        plan.first_layer = 0;
        plan.last_layer = 31;
        plan.has_embedding = true;
        plan.has_lm_head = true;
        plan.tp_scope = TPScope::LOCAL;
        plan.local_tp_devices = {
            GlobalDeviceAddress::cuda(0), // NVIDIA
            GlobalDeviceAddress::rocm(0)  // AMD
        };
        plan.local_tp_weights = {0.73f, 0.27f};
        plan.weight_shard.shard_index = 0;
        plan.weight_shard.total_shards = 2;
        plan.weight_shard.work_fraction = 0.73f;
        return plan;
    }
};

TEST_F(Test__Qwen2GraphConfigBuilder_LocalTP, TwoDevices_EqualSplit)
{
    auto plan = createTwoDeviceTPPlan();
    auto result = builder->buildConfig(plan, model_config,
                                       g_stub_weight_manager);

    ASSERT_TRUE(result.success);
    EXPECT_EQ(result.execution_info.total_shards, 2);
}

TEST_F(Test__Qwen2GraphConfigBuilder_LocalTP, TwoDevices_HeadDistribution)
{
    auto plan = createTwoDeviceTPPlan();
    auto result = builder->buildConfig(plan, model_config,
                                       g_stub_weight_manager);

    ASSERT_TRUE(result.success);
    ASSERT_NE(result.placement, nullptr);

    // With 28 heads and 2 devices (equal weights), expect 14 each
    int cuda0_heads = result.placement->headsForDevice(GlobalDeviceAddress::cuda(0));
    int cuda1_heads = result.placement->headsForDevice(GlobalDeviceAddress::cuda(1));

    EXPECT_EQ(cuda0_heads + cuda1_heads, model_config.n_heads);
    EXPECT_EQ(cuda0_heads, 14);
    EXPECT_EQ(cuda1_heads, 14);
}

TEST_F(Test__Qwen2GraphConfigBuilder_LocalTP, ProportionalWeights)
{
    auto plan = createProportionalTPPlan();
    auto result = builder->buildConfig(plan, model_config,
                                       g_stub_weight_manager);

    ASSERT_TRUE(result.success);
    ASSERT_NE(result.placement, nullptr);

    // With 28 heads and 73%/27% weights
    int cuda_heads = result.placement->headsForDevice(GlobalDeviceAddress::cuda(0));
    int rocm_heads = result.placement->headsForDevice(GlobalDeviceAddress::rocm(0));

    EXPECT_EQ(cuda_heads + rocm_heads, model_config.n_heads);
    EXPECT_GT(cuda_heads, rocm_heads);

    // Approximately 73% for CUDA
    float cuda_fraction = static_cast<float>(cuda_heads) / model_config.n_heads;
    EXPECT_NEAR(cuda_fraction, 0.73f, 0.1f);
}

TEST_F(Test__Qwen2GraphConfigBuilder_LocalTP, AllDevicesBuildAllLayers)
{
    auto plan = createTwoDeviceTPPlan();
    auto result = builder->buildConfig(plan, model_config,
                                       g_stub_weight_manager);

    ASSERT_TRUE(result.success);
    ASSERT_NE(result.placement, nullptr);

    // In local TP, all devices work on all layers
    for (int i = 0; i < model_config.n_layers; ++i)
    {
        EXPECT_TRUE(result.placement->shouldBuildLayer(i))
            << "Local TP should build layer " << i;
    }
}

// ============================================================================
// Combined PP + TP Tests
// ============================================================================

TEST(Test__Qwen2GraphConfigBuilder_Combined, PP_WithTP)
{
    auto builder = createQwen2GraphConfigBuilder();
    auto model_config = createTestModelConfig();

    // First PP stage with 2-way TP
    RankExecutionPlan plan;
    plan.rank = 0;
    plan.hostname = "localhost";
    plan.numa_node = 0;
    plan.primary_device = GlobalDeviceAddress::cuda(0);
    plan.first_layer = 0;
    plan.last_layer = 15;
    plan.has_embedding = true;
    plan.has_lm_head = false;
    plan.next_rank = 1;
    plan.local_tp_devices = {
        GlobalDeviceAddress::cuda(0),
        GlobalDeviceAddress::cuda(1)};
    plan.local_tp_weights = {0.5f, 0.5f};
    plan.weight_shard.shard_index = 0;
    plan.weight_shard.total_shards = 2;
    plan.weight_shard.work_fraction = 0.5f;

    auto result = builder->buildConfig(plan, model_config,
                                       g_stub_weight_manager);

    ASSERT_TRUE(result.success);
    EXPECT_EQ(result.execution_info.first_layer, 0);
    EXPECT_EQ(result.execution_info.last_layer, 15);
    EXPECT_TRUE(result.execution_info.has_embedding);
    EXPECT_FALSE(result.execution_info.has_lm_head);
    EXPECT_EQ(result.execution_info.total_shards, 2);
}

// ============================================================================
// Edge Cases
// ============================================================================

TEST(Test__Qwen2GraphConfigBuilder_EdgeCases, SingleLayer)
{
    auto builder = createQwen2GraphConfigBuilder();
    ModelConfig model_config = createTestModelConfig();
    model_config.n_layers = 1;

    RankExecutionPlan plan = createSingleDevicePlan();
    plan.first_layer = 0;
    plan.last_layer = 0;

    auto result = builder->buildConfig(plan, model_config,
                                       g_stub_weight_manager);

    ASSERT_TRUE(result.success);
    EXPECT_EQ(result.model_info.n_layers, 1);
}

TEST(Test__Qwen2GraphConfigBuilder_EdgeCases, ManyShards)
{
    auto builder = createQwen2GraphConfigBuilder();
    ModelConfig model_config = createTestModelConfig();

    // 8-way TP with equal weights
    RankExecutionPlan plan;
    plan.rank = 0;
    plan.hostname = "localhost";
    plan.numa_node = 0;
    plan.primary_device = GlobalDeviceAddress::cuda(0);
    plan.first_layer = 0;
    plan.last_layer = 31;
    plan.has_embedding = true;
    plan.has_lm_head = true;

    std::vector<GlobalDeviceAddress> devices;
    std::vector<float> weights;
    for (int i = 0; i < 8; ++i)
    {
        devices.push_back(GlobalDeviceAddress::cuda(i));
        weights.push_back(0.125f);
    }
    plan.local_tp_devices = devices;
    plan.local_tp_weights = weights;
    plan.weight_shard.shard_index = 0;
    plan.weight_shard.total_shards = 8;
    plan.weight_shard.work_fraction = 0.125f;

    auto result = builder->buildConfig(plan, model_config,
                                       g_stub_weight_manager);

    ASSERT_TRUE(result.success);
    EXPECT_EQ(result.execution_info.total_shards, 8);
}
