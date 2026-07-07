/**
 * @file Test__DeviceGraphOrchestratorDepsConstruction.cpp
 * @brief Unit tests for DeviceGraphOrchestrator Dependencies-based construction
 *
 * Tests the preferred Dependencies struct constructor path introduced during
 * DGO-Qwen2 decoupling (Phase 5). Verifies:
 * - Required fields validation (model_ctx, graph_builder)
 * - Optional field wiring (topology, collective_ctx, pp_stage_config, etc.)
 * - Polymorphic IGraphBuilder usage (MockGraphBuilder, not QwenStandardGraph)
 * - PP stage config validation during construction
 * - Field accessibility after construction
 *
 * @author David Sanftenberg
 * @date April 2026
 */

#include <gtest/gtest.h>
#include <cstdlib>
#include <initializer_list>
#include <memory>
#include <string>
#include <vector>

#include "execution/local_execution/orchestrators/DeviceGraphOrchestrator.h"
#include "execution/local_execution/graph/IGraphBuilder.h"
#include "execution/local_execution/collective/CollectiveContext.h"
#include "execution/factory/FactoryPPStageConfig.h"
#include "config/PipelineConfig.h"
#include "config/TensorParallelConfig.h"
#include "backends/DeviceId.h"
#include "utils/DebugEnv.h"
#include "../../../../mocks/MockLocalTPContext.h"
#include "../../../../mocks/MockModelContext.h"
#include "../../../../mocks/MockComputeStage.h"

using namespace llaminar2;

namespace
{
    class ScopedEnvVars
    {
    public:
        explicit ScopedEnvVars(std::initializer_list<std::pair<const char *, const char *>> values)
        {
            for (const auto &[name, value] : values)
            {
                Entry entry;
                entry.name = name;
                if (const char *old_value = std::getenv(name))
                {
                    entry.had_value = true;
                    entry.old_value = old_value;
                }
                entries_.push_back(entry);
                if (value)
                    ::setenv(name, value, 1);
                else
                    ::unsetenv(name);
            }
            mutableDebugEnv().reload();
        }

        ~ScopedEnvVars()
        {
            for (const auto &entry : entries_)
            {
                if (entry.had_value)
                    ::setenv(entry.name.c_str(), entry.old_value.c_str(), 1);
                else
                    ::unsetenv(entry.name.c_str());
            }
            mutableDebugEnv().reload();
        }

        ScopedEnvVars(const ScopedEnvVars &) = delete;
        ScopedEnvVars &operator=(const ScopedEnvVars &) = delete;

    private:
        struct Entry
        {
            std::string name;
            bool had_value = false;
            std::string old_value;
        };

        std::vector<Entry> entries_;
    };
} // namespace

// =============================================================================
// Test Fixture
// =============================================================================

/**
 * @brief Test fixture for Dependencies-based DGO construction.
 *
 * Uses MockGraphBuilder (from IGraphBuilder.h) to test DGO construction
 * without coupling to any specific model architecture.
 */
class Test__DeviceGraphOrchestratorDepsConstruction : public ::testing::Test
{
protected:
    void SetUp() override
    {
        DeviceManager::instance().initialize(-1);

        // Use MockGraphBuilder — verifies DGO works with IGraphBuilder interface,
        // not just QwenStandardGraph.
        mock_builder_ = std::make_shared<MockGraphBuilder>();
        mock_builder_->setNumLayers(24);
        mock_builder_->setHiddenDim(896);

        GraphConfig cfg{};
        cfg.n_layers = 24;
        cfg.d_model = 896;
        cfg.n_heads = 14;
        cfg.n_kv_heads = 2;
        cfg.head_dim = 64;
        cfg.d_ff = 4864;
        cfg.vocab_size = 151936;
        cfg.default_device = DeviceId::cpu();
        mock_builder_->setConfig(cfg);

        // MockModelContext — required by Dependencies constructor
        mock_model_ctx_ = llaminar2::test::MockModelContext::createQwen2_05B();
    }

    /**
     * @brief Build minimal Dependencies with only required fields.
     */
    DeviceGraphOrchestrator::Dependencies minimalDeps()
    {
        DeviceGraphOrchestrator::Dependencies deps;
        deps.graph_builder = mock_builder_;
        deps.model_ctx = mock_model_ctx_;
        return deps;
    }

    std::shared_ptr<MockGraphBuilder> mock_builder_;
    std::shared_ptr<llaminar2::test::MockModelContext> mock_model_ctx_;
};

// =============================================================================
// Required Fields — graph_builder
// =============================================================================

TEST_F(Test__DeviceGraphOrchestratorDepsConstruction, MinimalDeps_OnlyGraphBuilder)
{
    auto deps = minimalDeps();
    DeviceGraphOrchestrator dgo(std::move(deps));

    EXPECT_NE(std::as_const(dgo).graphBuilder(), nullptr);
    EXPECT_EQ(std::as_const(dgo).graphBuilder(), mock_builder_.get());
    EXPECT_TRUE(dgo.isGraphCachingEnabled()); // Default cache_config: enabled=true
}

TEST_F(Test__DeviceGraphOrchestratorDepsConstruction, NullGraphBuilder_Throws)
{
    DeviceGraphOrchestrator::Dependencies deps;
    deps.model_ctx = mock_model_ctx_;
    deps.graph_builder = nullptr;

    EXPECT_THROW(DeviceGraphOrchestrator(std::move(deps)), std::invalid_argument);
}

TEST_F(Test__DeviceGraphOrchestratorDepsConstruction, NullModelCtx_Throws)
{
    DeviceGraphOrchestrator::Dependencies deps;
    deps.graph_builder = mock_builder_;
    deps.model_ctx = nullptr;

    EXPECT_THROW(DeviceGraphOrchestrator(std::move(deps)), std::invalid_argument);
}

// =============================================================================
// Polymorphic IGraphBuilder Usage (MockGraphBuilder)
// =============================================================================

TEST_F(Test__DeviceGraphOrchestratorDepsConstruction, MockGraphBuilder_PropertiesAccessible)
{
    auto deps = minimalDeps();
    DeviceGraphOrchestrator dgo(std::move(deps));

    // Access goes through IGraphBuilder* pointer — validates polymorphism
    const IGraphBuilder *builder = std::as_const(dgo).graphBuilder();
    EXPECT_EQ(builder->numLayers(), 24);
    EXPECT_EQ(builder->hiddenDim(), 896);
    EXPECT_TRUE(builder->isInitialized());
}

TEST_F(Test__DeviceGraphOrchestratorDepsConstruction, MockGraphBuilder_ConfigAccessible)
{
    auto deps = minimalDeps();
    DeviceGraphOrchestrator dgo(std::move(deps));

    const auto &cfg = std::as_const(dgo).graphBuilder()->config();
    EXPECT_EQ(cfg.n_layers, 24);
    EXPECT_EQ(cfg.d_model, 896);
    EXPECT_EQ(cfg.n_heads, 14);
    EXPECT_EQ(cfg.head_dim, 64);
}

TEST_F(Test__DeviceGraphOrchestratorDepsConstruction, DecodeCapturePolicy_DisableCapturedCollectivesOverrideKeepsCollectiveDecodeEager)
{
    ScopedEnvVars env({
        {"LLAMINAR_GPU_GRAPHS", "1"},
        {"LLAMINAR_GPU_GRAPH_COLLECTIVE_SEGMENTED", "0"},
        {"LLAMINAR_GPU_GRAPH_CAPTURE_COLLECTIVES", "0"},
    });

    auto deps = minimalDeps();
    DeviceGraphOrchestrator dgo(std::move(deps));
    llaminar2::testing::MockDeviceContext gpu_ctx(DeviceId::rocm(0), ComputeBackendType::GPU_ROCM);
    const IForwardExecutionHost &host = dgo;

    const auto policy = host.buildDecodeCapturePolicy(
        true,
        &gpu_ctx,
        0);

    EXPECT_TRUE(policy.allow_fast_decode);
    EXPECT_FALSE(policy.collective_segmented_enabled);
    EXPECT_FALSE(policy.collectives_graph_capturable);
    EXPECT_FALSE(policy.allow_cached_graph_replay);
}

TEST_F(Test__DeviceGraphOrchestratorDepsConstruction, DeviceMoERebalanceGraphControllerDefaultsEnabled)
{
    ScopedEnvVars env({
        {"LLAMINAR_MOE_DEVICE_REBALANCE_GRAPH_CONTROLLER", nullptr},
    });

    EXPECT_TRUE(debugEnv().moe_rebalance.device_rebalance_graph_controller)
        << "Homogeneous GPU dynamic rebalance should use the graph-native device controller by default.";
}

TEST_F(Test__DeviceGraphOrchestratorDepsConstruction, DeviceMoERebalanceGraphControllerEnvCanDisableDefault)
{
    ScopedEnvVars env({
        {"LLAMINAR_MOE_DEVICE_REBALANCE_GRAPH_CONTROLLER", "0"},
    });

    EXPECT_FALSE(debugEnv().moe_rebalance.device_rebalance_graph_controller);
}

TEST_F(Test__DeviceGraphOrchestratorDepsConstruction, DeviceMoERebalancePayloadSidebandDefaultsOff)
{
    ScopedEnvVars env({
        {"LLAMINAR_MOE_DEVICE_REBALANCE_PAYLOAD_SIDEBAND", nullptr},
    });

    EXPECT_FALSE(debugEnv().moe_rebalance.device_rebalance_payload_sideband)
        << "Bulk expert payload sidebands must not be attached to every decode-token graph replay by default.";
}

TEST_F(Test__DeviceGraphOrchestratorDepsConstruction, DeviceMoERebalancePayloadSidebandEnvStillParsesButSelectorRejects)
{
    ScopedEnvVars env({
        {"LLAMINAR_MOE_DEVICE_REBALANCE_PAYLOAD_SIDEBAND", "1"},
    });

    EXPECT_TRUE(debugEnv().moe_rebalance.device_rebalance_payload_sideband);
}

TEST_F(Test__DeviceGraphOrchestratorDepsConstruction, DeviceMoERebalanceLayerWaveDefaultsToBoundedEarlySweep)
{
    ScopedEnvVars env({
        {"LLAMINAR_MOE_DEVICE_REBALANCE_LAYER_WAVE", nullptr},
    });

    EXPECT_EQ(debugEnv().moe_rebalance.device_rebalance_layer_wave_count, 4)
        << "The graph-side controller should inspect several early layers per maintenance wave "
           "without defaulting to a full-model policy pass.";
}

TEST_F(Test__DeviceGraphOrchestratorDepsConstruction, DeviceMoERebalanceLayerWaveEnvOverridesDefault)
{
    ScopedEnvVars env({
        {"LLAMINAR_MOE_DEVICE_REBALANCE_LAYER_WAVE", "7"},
    });

    EXPECT_EQ(debugEnv().moe_rebalance.device_rebalance_layer_wave_count, 7);
}

TEST_F(Test__DeviceGraphOrchestratorDepsConstruction, DeviceMoERebalanceLoadSpreadFloorDefaultsToDynamic)
{
    ScopedEnvVars env({
        {"LLAMINAR_MOE_DEVICE_REBALANCE_MIN_LOAD_SPREAD_IMPROVEMENT", nullptr},
        {"LLAMINAR_MOE_DEVICE_REBALANCE_MIN_LOAD_SPREAD_IMPROVEMENT_DIVISOR", nullptr},
        {"LLAMINAR_MOE_DEVICE_REBALANCE_MIN_WAVE_SPREAD_IMPROVEMENT_PER_PAYLOAD_SLOT", nullptr},
        {"LLAMINAR_MOE_DEVICE_REBALANCE_MIN_FOREIGN_ROWS_PER_TRANSFER", nullptr},
        {"LLAMINAR_MOE_DEVICE_REBALANCE_MIN_ROUTER_SPREAD_IMPROVEMENT_PER_PAYLOAD_SLOT", nullptr},
        {"LLAMINAR_MOE_DEVICE_REBALANCE_MAX_POST_WAVE_LOAD_SPREAD_PERMILLE", nullptr},
        {"LLAMINAR_MOE_DEVICE_REBALANCE_NO_WORK_BACKOFF_PERIODS", nullptr},
    });

    EXPECT_EQ(debugEnv().moe_rebalance.device_rebalance_min_load_spread_improvement, 0)
        << "Default Dynamic should use the shared Dynamic admission floor "
           "max(2, window/16), not an extra GPU-only candidate floor.";
    EXPECT_EQ(debugEnv().moe_rebalance.device_rebalance_min_load_spread_improvement_divisor,
              static_cast<int>(moe_rebalance_policy::kDefaultDeviceMinLoadSpreadImprovementDivisor))
        << "The relative spread gate is enabled by default to keep transfer-backed GPU "
           "maintenance economical.";
    EXPECT_EQ(debugEnv().moe_rebalance.device_rebalance_min_wave_spread_improvement_per_payload_slot, 256)
        << "The wave-level value gate should reject low-value hot-cache transfer churn by default.";
    EXPECT_EQ(debugEnv().moe_rebalance.device_rebalance_min_foreign_rows_per_transfer, 0)
        << "The useful-work gate should be available for sweeps without changing existing policy by default.";
    EXPECT_EQ(debugEnv().moe_rebalance.device_rebalance_min_router_spread_improvement_per_payload_slot, 128)
        << "Steady-state hot-replica transfer waves should require measured realized router benefit by default.";
    EXPECT_EQ(debugEnv().moe_rebalance.device_rebalance_max_post_wave_load_spread_per_mille, 100)
        << "Transfer-backed waves should leave the domain close enough to balanced to amortize maintenance cost.";
    EXPECT_EQ(debugEnv().moe_rebalance.device_rebalance_no_work_backoff_periods, 1)
        << "Empty maintenance replays should back off by default until zero-bucket graph bodies exist.";
}

TEST_F(Test__DeviceGraphOrchestratorDepsConstruction, DeviceMoERebalanceLoadSpreadFloorEnvOverridesDefault)
{
    ScopedEnvVars env({
        {"LLAMINAR_MOE_DEVICE_REBALANCE_MIN_LOAD_SPREAD_IMPROVEMENT", "96"},
        {"LLAMINAR_MOE_DEVICE_REBALANCE_MIN_LOAD_SPREAD_IMPROVEMENT_DIVISOR", "12"},
        {"LLAMINAR_MOE_DEVICE_REBALANCE_MIN_WAVE_SPREAD_IMPROVEMENT_PER_PAYLOAD_SLOT", "192"},
        {"LLAMINAR_MOE_DEVICE_REBALANCE_MIN_FOREIGN_ROWS_PER_TRANSFER", "768"},
        {"LLAMINAR_MOE_DEVICE_REBALANCE_MIN_ROUTER_SPREAD_IMPROVEMENT_PER_PAYLOAD_SLOT", "384"},
        {"LLAMINAR_MOE_DEVICE_REBALANCE_MAX_POST_WAVE_LOAD_SPREAD_PERMILLE", "75"},
        {"LLAMINAR_MOE_DEVICE_REBALANCE_NO_WORK_BACKOFF_PERIODS", "3"},
    });

    EXPECT_EQ(debugEnv().moe_rebalance.device_rebalance_min_load_spread_improvement, 96);
    EXPECT_EQ(debugEnv().moe_rebalance.device_rebalance_min_load_spread_improvement_divisor, 12);
    EXPECT_TRUE(debugEnv().presence.has("LLAMINAR_MOE_DEVICE_REBALANCE_MIN_LOAD_SPREAD_IMPROVEMENT_DIVISOR"));
    EXPECT_EQ(debugEnv().moe_rebalance.device_rebalance_min_wave_spread_improvement_per_payload_slot, 192);
    EXPECT_EQ(debugEnv().moe_rebalance.device_rebalance_min_foreign_rows_per_transfer, 768);
    EXPECT_TRUE(debugEnv().presence.has("LLAMINAR_MOE_DEVICE_REBALANCE_MIN_FOREIGN_ROWS_PER_TRANSFER"));
    EXPECT_EQ(debugEnv().moe_rebalance.device_rebalance_min_router_spread_improvement_per_payload_slot, 384);
    EXPECT_EQ(debugEnv().moe_rebalance.device_rebalance_max_post_wave_load_spread_per_mille, 75);
    EXPECT_EQ(debugEnv().moe_rebalance.device_rebalance_no_work_backoff_periods, 3);
}

TEST_F(Test__DeviceGraphOrchestratorDepsConstruction, DecodeCapturePolicy_CapturesCollectivesByDefaultForHomogeneousCudaLocalTP)
{
    ScopedEnvVars env({
        {"LLAMINAR_GPU_GRAPHS", "1"},
        {"LLAMINAR_GPU_GRAPH_COLLECTIVE_SEGMENTED", nullptr},
        {"LLAMINAR_GPU_GRAPH_CAPTURE_COLLECTIVES", nullptr},
    });

    auto tp_ctx = std::make_shared<llaminar2::test::MockLocalTPContext>();
    tp_ctx->setBackend(CollectiveBackendType::NCCL);
    tp_ctx->setDevices({GlobalDeviceAddress::cuda(0), GlobalDeviceAddress::cuda(1)});

    GraphConfig cfg = mock_builder_->config();
    cfg.tp_ctx = tp_ctx.get();
    mock_builder_->setConfig(cfg);

    auto deps = minimalDeps();
    DeviceGraphOrchestrator dgo(std::move(deps));
    llaminar2::testing::MockDeviceContext gpu_ctx(DeviceId::cuda(0), ComputeBackendType::GPU_CUDA);
    const IForwardExecutionHost &host = dgo;

    const auto policy = host.buildDecodeCapturePolicy(
        true,
        &gpu_ctx,
        0);

    EXPECT_TRUE(policy.allow_fast_decode);
    EXPECT_FALSE(policy.collective_segmented_enabled);
    EXPECT_TRUE(policy.collectives_graph_capturable);
    EXPECT_TRUE(policy.allow_cached_graph_replay)
        << "Homogeneous CUDA LocalTP collective decode must remain graph-replay eligible by default.";
}

TEST_F(Test__DeviceGraphOrchestratorDepsConstruction, DecodeCapturePolicy_CapturesCollectivesByDefaultForHomogeneousRocmLocalTP)
{
    ScopedEnvVars env({
        {"LLAMINAR_GPU_GRAPHS", "1"},
        {"LLAMINAR_GPU_GRAPH_COLLECTIVE_SEGMENTED", nullptr},
        {"LLAMINAR_GPU_GRAPH_CAPTURE_COLLECTIVES", nullptr},
    });

    auto tp_ctx = std::make_shared<llaminar2::test::MockLocalTPContext>();
    tp_ctx->setBackend(CollectiveBackendType::RCCL);
    tp_ctx->setDevices({GlobalDeviceAddress::rocm(0), GlobalDeviceAddress::rocm(1)});

    GraphConfig cfg = mock_builder_->config();
    cfg.tp_ctx = tp_ctx.get();
    mock_builder_->setConfig(cfg);

    auto deps = minimalDeps();
    DeviceGraphOrchestrator dgo(std::move(deps));
    llaminar2::testing::MockDeviceContext gpu_ctx(DeviceId::rocm(0), ComputeBackendType::GPU_ROCM);
    const IForwardExecutionHost &host = dgo;

    const auto policy = host.buildDecodeCapturePolicy(
        true,
        &gpu_ctx,
        0);

    EXPECT_TRUE(policy.allow_fast_decode);
    EXPECT_FALSE(policy.collective_segmented_enabled);
    EXPECT_TRUE(policy.collectives_graph_capturable);
    EXPECT_TRUE(policy.allow_cached_graph_replay)
        << "Homogeneous ROCm LocalTP collective decode should graph-capture through participant-local RCCL enqueue.";
}

TEST_F(Test__DeviceGraphOrchestratorDepsConstruction, DecodeCapturePolicy_CapturesDenseDecodeReplicatedWithCollectiveOptIn)
{
    ScopedEnvVars env({
        {"LLAMINAR_GPU_GRAPHS", "1"},
        {"LLAMINAR_GPU_GRAPH_COLLECTIVE_SEGMENTED", "0"},
        {"LLAMINAR_GPU_GRAPH_CAPTURE_COLLECTIVES", "1"},
    });

    auto tp_ctx = std::make_shared<llaminar2::test::MockLocalTPContext>();
    tp_ctx->setBackend(CollectiveBackendType::NCCL);
    tp_ctx->setDevices({GlobalDeviceAddress::cuda(0), GlobalDeviceAddress::cuda(1)});

    GraphConfig cfg = mock_builder_->config();
    cfg.tp_ctx = tp_ctx.get();
    cfg.dense_tp_enabled = true;
    cfg.dense_tp_decode_replicated = true;
    mock_builder_->setConfig(cfg);

    auto deps = minimalDeps();
    DeviceGraphOrchestrator dgo(std::move(deps));
    llaminar2::testing::MockDeviceContext gpu_ctx(DeviceId::cuda(0), ComputeBackendType::GPU_CUDA);
    const IForwardExecutionHost &host = dgo;

    const auto policy = host.buildDecodeCapturePolicy(
        true,
        &gpu_ctx,
        0);

    EXPECT_TRUE(policy.allow_fast_decode);
    EXPECT_FALSE(policy.collective_segmented_enabled);
    EXPECT_TRUE(policy.collectives_graph_capturable);
    EXPECT_TRUE(policy.allow_cached_graph_replay)
        << "Phase-split dense decode builds a decode-only graph with replicated dense bindings; "
           "homogeneous LocalTP captured collectives remain the gating contract.";
}

TEST_F(Test__DeviceGraphOrchestratorDepsConstruction, DecodeCapturePolicy_CapturesCollectivesForOptInHomogeneousCudaLocalTP)
{
    ScopedEnvVars env({
        {"LLAMINAR_GPU_GRAPHS", "1"},
        {"LLAMINAR_GPU_GRAPH_COLLECTIVE_SEGMENTED", "0"},
        {"LLAMINAR_GPU_GRAPH_CAPTURE_COLLECTIVES", "1"},
    });

    auto tp_ctx = std::make_shared<llaminar2::test::MockLocalTPContext>();
    tp_ctx->setBackend(CollectiveBackendType::NCCL);
    tp_ctx->setDevices({GlobalDeviceAddress::cuda(0), GlobalDeviceAddress::cuda(1)});

    GraphConfig cfg = mock_builder_->config();
    cfg.tp_ctx = tp_ctx.get();
    mock_builder_->setConfig(cfg);

    auto deps = minimalDeps();
    DeviceGraphOrchestrator dgo(std::move(deps));
    llaminar2::testing::MockDeviceContext gpu_ctx(DeviceId::cuda(0), ComputeBackendType::GPU_CUDA);
    const IForwardExecutionHost &host = dgo;

    const auto policy = host.buildDecodeCapturePolicy(
        true,
        &gpu_ctx,
        0);

    EXPECT_TRUE(policy.allow_fast_decode);
    EXPECT_FALSE(policy.collective_segmented_enabled);
    EXPECT_TRUE(policy.collectives_graph_capturable);
    EXPECT_TRUE(policy.allow_cached_graph_replay);
}

TEST_F(Test__DeviceGraphOrchestratorDepsConstruction, DecodeCapturePolicy_CapturesCollectivesForOptInHomogeneousRocmLocalTP)
{
    ScopedEnvVars env({
        {"LLAMINAR_GPU_GRAPHS", "1"},
        {"LLAMINAR_GPU_GRAPH_COLLECTIVE_SEGMENTED", "0"},
        {"LLAMINAR_GPU_GRAPH_CAPTURE_COLLECTIVES", "1"},
    });

    auto tp_ctx = std::make_shared<llaminar2::test::MockLocalTPContext>();
    tp_ctx->setBackend(CollectiveBackendType::RCCL);
    tp_ctx->setDevices({GlobalDeviceAddress::rocm(0), GlobalDeviceAddress::rocm(1)});

    GraphConfig cfg = mock_builder_->config();
    cfg.tp_ctx = tp_ctx.get();
    mock_builder_->setConfig(cfg);

    auto deps = minimalDeps();
    DeviceGraphOrchestrator dgo(std::move(deps));
    llaminar2::testing::MockDeviceContext gpu_ctx(DeviceId::rocm(0), ComputeBackendType::GPU_ROCM);
    const IForwardExecutionHost &host = dgo;

    const auto policy = host.buildDecodeCapturePolicy(
        true,
        &gpu_ctx,
        0);

    EXPECT_TRUE(policy.allow_fast_decode);
    EXPECT_FALSE(policy.collective_segmented_enabled);
    EXPECT_TRUE(policy.collectives_graph_capturable);
    EXPECT_TRUE(policy.allow_cached_graph_replay);
}

TEST_F(Test__DeviceGraphOrchestratorDepsConstruction, DecodeCapturePolicy_RejectsCapturedCollectivesForMixedLocalTP)
{
    ScopedEnvVars env({
        {"LLAMINAR_GPU_GRAPHS", "1"},
        {"LLAMINAR_GPU_GRAPH_COLLECTIVE_SEGMENTED", "0"},
        {"LLAMINAR_GPU_GRAPH_CAPTURE_COLLECTIVES", "1"},
    });

    auto tp_ctx = std::make_shared<llaminar2::test::MockLocalTPContext>();
    tp_ctx->setBackend(CollectiveBackendType::NCCL);
    tp_ctx->setDevices({GlobalDeviceAddress::cuda(0), GlobalDeviceAddress::rocm(0)});

    GraphConfig cfg = mock_builder_->config();
    cfg.tp_ctx = tp_ctx.get();
    mock_builder_->setConfig(cfg);

    auto deps = minimalDeps();
    DeviceGraphOrchestrator dgo(std::move(deps));
    llaminar2::testing::MockDeviceContext gpu_ctx(DeviceId::cuda(0), ComputeBackendType::GPU_CUDA);
    const IForwardExecutionHost &host = dgo;

    const auto policy = host.buildDecodeCapturePolicy(
        true,
        &gpu_ctx,
        0);

    EXPECT_TRUE(policy.allow_fast_decode);
    EXPECT_FALSE(policy.collective_segmented_enabled);
    EXPECT_FALSE(policy.collectives_graph_capturable);
    EXPECT_FALSE(policy.allow_cached_graph_replay);
}

TEST_F(Test__DeviceGraphOrchestratorDepsConstruction, ExecutorAccessible)
{
    auto deps = minimalDeps();
    DeviceGraphOrchestrator dgo(std::move(deps));

    // Executor should be accessible (default-constructed)
    const DeviceGraphExecutor &exec = std::as_const(dgo).executor();
    (void)exec;
}

// =============================================================================
// Optional: Graph Cache Config
// =============================================================================

TEST_F(Test__DeviceGraphOrchestratorDepsConstruction, CacheConfig_EnabledByDefault)
{
    auto deps = minimalDeps();
    DeviceGraphOrchestrator dgo(std::move(deps));

    EXPECT_TRUE(dgo.isGraphCachingEnabled());
}

TEST_F(Test__DeviceGraphOrchestratorDepsConstruction, CacheConfig_DisabledViaField)
{
    auto deps = minimalDeps();
    deps.cache_config.enabled = false;
    DeviceGraphOrchestrator dgo(std::move(deps));

    EXPECT_FALSE(dgo.isGraphCachingEnabled());
}

TEST_F(Test__DeviceGraphOrchestratorDepsConstruction, CacheConfig_CustomDecodeSeqLen)
{
    auto deps = minimalDeps();
    deps.cache_config.decode_seq_len = 4;
    deps.cache_config.cache_attention = false;

    // Construction should succeed with custom caching params
    DeviceGraphOrchestrator dgo(std::move(deps));
    EXPECT_TRUE(dgo.isGraphCachingEnabled());
}

// =============================================================================
// Optional: CollectiveContext
// =============================================================================

TEST_F(Test__DeviceGraphOrchestratorDepsConstruction, CollectiveCtx_NullByDefault)
{
    auto deps = minimalDeps();
    DeviceGraphOrchestrator dgo(std::move(deps));

    EXPECT_EQ(dgo.collectiveContext(), nullptr);
    EXPECT_FALSE(dgo.isGpuCollectivesEnabled());
}

TEST_F(Test__DeviceGraphOrchestratorDepsConstruction, CollectiveCtx_WiredFromDeps)
{
    auto deps = minimalDeps();
    deps.collective_ctx = CollectiveContextFactory::createSingleDevice();
    auto *raw = deps.collective_ctx.get();

    DeviceGraphOrchestrator dgo(std::move(deps));

    ASSERT_NE(dgo.collectiveContext(), nullptr);
    EXPECT_EQ(dgo.collectiveContext().get(), raw);
    EXPECT_TRUE(dgo.isGpuCollectivesEnabled());
}

// =============================================================================
// Optional: PP Stage Config
// =============================================================================

TEST_F(Test__DeviceGraphOrchestratorDepsConstruction, PPStageConfig_NotSetByDefault)
{
    auto deps = minimalDeps();
    DeviceGraphOrchestrator dgo(std::move(deps));

    // PP stage config is optional — when absent, full model is assumed
    // (no partial layer graph building)
    EXPECT_NE(std::as_const(dgo).graphBuilder(), nullptr);
}

TEST_F(Test__DeviceGraphOrchestratorDepsConstruction, PPStageConfig_ValidFirstStage)
{
    auto deps = minimalDeps();
    deps.pp_stage_config = FactoryPPStageConfig{
        .first_layer = 0,
        .last_layer = 12,
        .has_embedding = true,
        .has_lm_head = false};

    // Should succeed — valid PP config
    DeviceGraphOrchestrator dgo(std::move(deps));
    EXPECT_NE(std::as_const(dgo).graphBuilder(), nullptr);
}

TEST_F(Test__DeviceGraphOrchestratorDepsConstruction, PPStageConfig_ValidLastStage)
{
    auto deps = minimalDeps();
    deps.pp_stage_config = FactoryPPStageConfig{
        .first_layer = 12,
        .last_layer = 24,
        .has_embedding = false,
        .has_lm_head = true};

    DeviceGraphOrchestrator dgo(std::move(deps));
    EXPECT_NE(std::as_const(dgo).graphBuilder(), nullptr);
}

TEST_F(Test__DeviceGraphOrchestratorDepsConstruction, PPStageConfig_ValidMiddleStage)
{
    auto deps = minimalDeps();
    deps.pp_stage_config = FactoryPPStageConfig{
        .first_layer = 8,
        .last_layer = 16,
        .has_embedding = false,
        .has_lm_head = false};

    DeviceGraphOrchestrator dgo(std::move(deps));
    EXPECT_NE(std::as_const(dgo).graphBuilder(), nullptr);
}

TEST_F(Test__DeviceGraphOrchestratorDepsConstruction, PPStageConfig_InvalidRange_Throws)
{
    auto deps = minimalDeps();
    deps.pp_stage_config = FactoryPPStageConfig{
        .first_layer = 10,
        .last_layer = 5, // Invalid: last < first
        .has_embedding = false,
        .has_lm_head = false};

    EXPECT_THROW(DeviceGraphOrchestrator(std::move(deps)), std::invalid_argument);
}

TEST_F(Test__DeviceGraphOrchestratorDepsConstruction, PPStageConfig_ZeroLayers_Throws)
{
    auto deps = minimalDeps();
    deps.pp_stage_config = FactoryPPStageConfig{
        .first_layer = 5,
        .last_layer = 5, // Invalid: zero layers
        .has_embedding = false,
        .has_lm_head = false};

    EXPECT_THROW(DeviceGraphOrchestrator(std::move(deps)), std::invalid_argument);
}

TEST_F(Test__DeviceGraphOrchestratorDepsConstruction, PPStageConfig_NegativeFirstLayer_Throws)
{
    auto deps = minimalDeps();
    deps.pp_stage_config = FactoryPPStageConfig{
        .first_layer = -1,
        .last_layer = 10,
        .has_embedding = false,
        .has_lm_head = false};

    EXPECT_THROW(DeviceGraphOrchestrator(std::move(deps)), std::invalid_argument);
}

// =============================================================================
// Optional: Pipeline Config
// =============================================================================

TEST_F(Test__DeviceGraphOrchestratorDepsConstruction, PipelineConfig_NullByDefault)
{
    auto deps = minimalDeps();
    DeviceGraphOrchestrator dgo(std::move(deps));

    // No pipeline config → single-device mode, no PP graph building
    EXPECT_NE(std::as_const(dgo).graphBuilder(), nullptr);
}

TEST_F(Test__DeviceGraphOrchestratorDepsConstruction, PipelineConfig_WiredFromDeps)
{
    auto deps = minimalDeps();

    auto pipeline = std::make_shared<PipelineConfig>();
    pipeline->total_layers = 24;
    pipeline->tp_domains = {
        {"gpu_a", {DeviceId::cuda(0)}, CollectiveBackendType::AUTO},
        {"gpu_b", {DeviceId::rocm(0)}, CollectiveBackendType::AUTO}};
    pipeline->pp_stages = {
        PPStageConfig::firstStage(0, "gpu_a", 0, 12),
        PPStageConfig::lastStage(1, "gpu_b", 12, 24)};

    deps.pipeline_config = pipeline;

    DeviceGraphOrchestrator dgo(std::move(deps));
    // Pipeline config is stored and will be used during graph building
    EXPECT_NE(std::as_const(dgo).graphBuilder(), nullptr);
}

// =============================================================================
// Optional: All Optional Fields Null — Succeeds
// =============================================================================

TEST_F(Test__DeviceGraphOrchestratorDepsConstruction, AllOptionalFieldsNull_Succeeds)
{
    // Explicitly verify that construction works with only required fields
    // and all optional fields at their default (null/empty) values.
    DeviceGraphOrchestrator::Dependencies deps;
    deps.graph_builder = mock_builder_;
    deps.model_ctx = mock_model_ctx_;
    // All other fields left at default (nullptr, nullopt, empty)

    DeviceGraphOrchestrator dgo(std::move(deps));

    EXPECT_NE(std::as_const(dgo).graphBuilder(), nullptr);
    EXPECT_EQ(dgo.collectiveContext(), nullptr);
    EXPECT_EQ(dgo.domainConfig(), nullptr);
    EXPECT_FALSE(dgo.isGpuCollectivesEnabled());
    // No crash — all optional fields are safely null
}

// =============================================================================
// Backward Compatibility: Legacy Constructor Still Works
// =============================================================================

TEST_F(Test__DeviceGraphOrchestratorDepsConstruction, LegacyConstructor_StillWorks)
{
    // The legacy (graph_builder, mpi_ctx) constructor is used by 87+ test sites.
    // Verify it still works after the Dependencies constructor was added.
    DeviceGraphOrchestrator dgo(mock_builder_, nullptr);

    EXPECT_NE(std::as_const(dgo).graphBuilder(), nullptr);
    EXPECT_EQ(std::as_const(dgo).graphBuilder(), mock_builder_.get());
    EXPECT_TRUE(dgo.isGraphCachingEnabled());
}

TEST_F(Test__DeviceGraphOrchestratorDepsConstruction, LegacyConstructor_WithCacheDisabled)
{
    GraphCacheConfig cache_config;
    cache_config.enabled = false;

    DeviceGraphOrchestrator dgo(mock_builder_, nullptr, cache_config);

    EXPECT_FALSE(dgo.isGraphCachingEnabled());
}

// =============================================================================
// setWeights / setBuffers delegation through IGraphBuilder
// =============================================================================

TEST_F(Test__DeviceGraphOrchestratorDepsConstruction, SetWeights_DelegatesToMockBuilder)
{
    auto deps = minimalDeps();
    DeviceGraphOrchestrator dgo(std::move(deps));

    ModelWeights weights{};
    std::unique_ptr<FP32Tensor> embed = std::make_unique<FP32Tensor>(
        std::vector<size_t>{151936, 896}, DeviceId::cpu());
    weights.embedding_table = embed.get();
    weights.get_layer_weights = [](int) -> LayerWeights
    { return {}; };

    dgo.setWeights(weights);

    // MockGraphBuilder's isInitialized should return true after setWeights
    EXPECT_TRUE(std::as_const(dgo).graphBuilder()->isInitialized());
}

TEST_F(Test__DeviceGraphOrchestratorDepsConstruction, SetBuffers_DelegatesToMockBuilder)
{
    auto deps = minimalDeps();
    DeviceGraphOrchestrator dgo(std::move(deps));

    ModelBuffers buffers{};
    EXPECT_NO_THROW(dgo.setBuffers(buffers));
}

// =============================================================================
// DomainConfig wiring via Dependencies
// =============================================================================

TEST_F(Test__DeviceGraphOrchestratorDepsConstruction, DomainConfig_NullByDefault)
{
    auto deps = minimalDeps();
    DeviceGraphOrchestrator dgo(std::move(deps));

    EXPECT_EQ(dgo.domainConfig(), nullptr);
}

TEST_F(Test__DeviceGraphOrchestratorDepsConstruction, DomainConfig_WiredFromDeps)
{
    auto deps = minimalDeps();

    TPDomain gpu_domain;
    gpu_domain.type = TPDomainType::GPU_INTRA_RANK;
    gpu_domain.name = "gpu_tp";
    gpu_domain.domain_size = 2;
    gpu_domain.local_rank_in_domain = 0;
    gpu_domain.devices = {DeviceId::cuda(0), DeviceId::rocm(0)};
    gpu_domain.communicator = MPI_COMM_NULL;

    deps.domain_config = std::make_shared<MultiDomainTPConfig>(
        MultiDomainTPConfig::createForTest({gpu_domain}));

    DeviceGraphOrchestrator dgo(std::move(deps));

    ASSERT_NE(dgo.domainConfig(), nullptr);
    EXPECT_EQ(dgo.domainConfig()->domains().size(), 1);
}

// =============================================================================
// Graph Cache Init with MockGraphBuilder
// =============================================================================

TEST_F(Test__DeviceGraphOrchestratorDepsConstruction, InitializeGraphCache_WithMockBuilder)
{
    auto deps = minimalDeps();
    DeviceGraphOrchestrator dgo(std::move(deps));

    dgo.initializeGraphCache(24);

    auto stats = dgo.getCacheStats();
    EXPECT_EQ(stats.cached_layers, 24u);
    EXPECT_EQ(stats.attention_cache_hits, 0u);
    EXPECT_EQ(stats.attention_cache_misses, 0u);
}

TEST_F(Test__DeviceGraphOrchestratorDepsConstruction, InvalidateExecutionCaches_WithMockBuilder)
{
    auto deps = minimalDeps();
    DeviceGraphOrchestrator dgo(std::move(deps));

    dgo.initializeGraphCache(24);
    dgo.invalidateExecutionCaches();

    auto stats = dgo.getCacheStats();
    EXPECT_EQ(stats.attention_cache_hits, 0u);
}

// =============================================================================
// Execute with empty graph + MockGraphBuilder
// =============================================================================

TEST_F(Test__DeviceGraphOrchestratorDepsConstruction, Execute_EmptyGraph_Succeeds)
{
    auto deps = minimalDeps();
    DeviceGraphOrchestrator dgo(std::move(deps));

    ComputeGraph graph;
    IDeviceContext *ctx = dgo.getDeviceContext(DeviceId::cpu());
    if (ctx == nullptr)
    {
        GTEST_SKIP() << "DeviceContext creation not available";
    }

    bool result = dgo.execute(graph, ctx);
    EXPECT_TRUE(result);
}
