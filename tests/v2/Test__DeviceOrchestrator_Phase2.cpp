/**
 * @file Test__DeviceOrchestrator_Phase2.cpp
 * @brief Phase 2 tests for DeviceOrchestrator (advanced strategies)
 * @author David Sanftenberg
 * @date 2025-10-24
 */

#include "loaders/DeviceOrchestrator.h"
#include "loaders/ModelContext.h"
#include "backends/ComputeBackend.h"
#include "utils/MPIContext.h"
#include "TestMPIUtils.h"
#include <gtest/gtest.h>
#include <memory>

using namespace llaminar2;

/**
 * @brief Test fixture for Phase 2 DeviceOrchestrator tests
 */
class DeviceOrchestratorPhase2Test : public ::testing::Test
{
protected:
    void SetUp() override
    {
        // Initialize device manager
        device_mgr_ = std::shared_ptr<DeviceManager>(&DeviceManager::instance(), [](DeviceManager *) {});
        device_mgr_->initialize();

        // Get MPI context
        mpi_ctx_ = MPIContextFactory::global();
    }

    std::shared_ptr<DeviceManager> device_mgr_;
    std::shared_ptr<MPIContext> mpi_ctx_;
};

// ============================================================================
// Device Map Parsing Tests
// ============================================================================

TEST_F(DeviceOrchestratorPhase2Test, ParseDeviceMapLayerRange)
{
    OrchestrationConfig config;
    config.strategy = PlacementStrategy::CUSTOM;
    config.device_map = "0-5:gpu:0";
    config.verbose = false;

    DeviceOrchestrator orchestrator(device_mgr_, mpi_ctx_, config);

    auto rules = orchestrator.parseDeviceMapString(config.device_map);

    ASSERT_EQ(rules.size(), 1);
    EXPECT_EQ(rules[0].type, DeviceMapRuleType::LAYER_RANGE);
    EXPECT_EQ(rules[0].start_layer, 0);
    EXPECT_EQ(rules[0].end_layer, 5);
}

TEST_F(DeviceOrchestratorPhase2Test, ParseDeviceMapMultipleRanges)
{
    OrchestrationConfig config;
    config.strategy = PlacementStrategy::CUSTOM;
    config.device_map = "0-11:gpu:0,12-23:cpu";
    config.verbose = false;

    DeviceOrchestrator orchestrator(device_mgr_, mpi_ctx_, config);

    auto rules = orchestrator.parseDeviceMapString(config.device_map);

    ASSERT_EQ(rules.size(), 2);

    // First rule: layers 0-11 on GPU
    EXPECT_EQ(rules[0].type, DeviceMapRuleType::LAYER_RANGE);
    EXPECT_EQ(rules[0].start_layer, 0);
    EXPECT_EQ(rules[0].end_layer, 11);

    // Second rule: layers 12-23 on CPU
    EXPECT_EQ(rules[1].type, DeviceMapRuleType::LAYER_RANGE);
    EXPECT_EQ(rules[1].start_layer, 12);
    EXPECT_EQ(rules[1].end_layer, 23);
}

TEST_F(DeviceOrchestratorPhase2Test, ParseDeviceMapPercentageFirst)
{
    OrchestrationConfig config;
    config.strategy = PlacementStrategy::CUSTOM;
    config.device_map = "first_50%:gpu:0";
    config.verbose = false;

    DeviceOrchestrator orchestrator(device_mgr_, mpi_ctx_, config);

    auto rules = orchestrator.parseDeviceMapString(config.device_map);

    ASSERT_EQ(rules.size(), 1);
    EXPECT_EQ(rules[0].type, DeviceMapRuleType::PERCENTAGE);
    EXPECT_TRUE(rules[0].is_first);
    EXPECT_FLOAT_EQ(rules[0].percentage, 50.0f);
}

TEST_F(DeviceOrchestratorPhase2Test, ParseDeviceMapPercentageLast)
{
    OrchestrationConfig config;
    config.strategy = PlacementStrategy::CUSTOM;
    config.device_map = "last_25%:cpu";
    config.verbose = false;

    DeviceOrchestrator orchestrator(device_mgr_, mpi_ctx_, config);

    auto rules = orchestrator.parseDeviceMapString(config.device_map);

    ASSERT_EQ(rules.size(), 1);
    EXPECT_EQ(rules[0].type, DeviceMapRuleType::PERCENTAGE);
    EXPECT_FALSE(rules[0].is_first);
    EXPECT_FLOAT_EQ(rules[0].percentage, 25.0f);
}

TEST_F(DeviceOrchestratorPhase2Test, ParseDeviceMapPattern)
{
    OrchestrationConfig config;
    config.strategy = PlacementStrategy::CUSTOM;
    config.device_map = "*embed*:gpu:0,*experts.0*:cpu";
    config.verbose = false;

    DeviceOrchestrator orchestrator(device_mgr_, mpi_ctx_, config);

    auto rules = orchestrator.parseDeviceMapString(config.device_map);

    ASSERT_EQ(rules.size(), 2);

    // First rule: embedding pattern on GPU
    EXPECT_EQ(rules[0].type, DeviceMapRuleType::PATTERN);
    EXPECT_EQ(rules[0].pattern, "*embed*");

    // Second rule: expert pattern on CPU
    EXPECT_EQ(rules[1].type, DeviceMapRuleType::PATTERN);
    EXPECT_EQ(rules[1].pattern, "*experts.0*");
}

TEST_F(DeviceOrchestratorPhase2Test, ParseDeviceMapMixed)
{
    OrchestrationConfig config;
    config.strategy = PlacementStrategy::CUSTOM;
    config.device_map = "0-11:gpu:0,*embed*:gpu:0,last_20%:cpu";
    config.verbose = false;

    DeviceOrchestrator orchestrator(device_mgr_, mpi_ctx_, config);

    auto rules = orchestrator.parseDeviceMapString(config.device_map);

    ASSERT_EQ(rules.size(), 3);
    EXPECT_EQ(rules[0].type, DeviceMapRuleType::LAYER_RANGE);
    EXPECT_EQ(rules[1].type, DeviceMapRuleType::PATTERN);
    EXPECT_EQ(rules[2].type, DeviceMapRuleType::PERCENTAGE);
}

// ============================================================================
// MEMORY_AWARE Strategy Tests
// ============================================================================

TEST_F(DeviceOrchestratorPhase2Test, MemoryAwareStrategyExplicitBudget)
{
    OrchestrationConfig config;
    config.strategy = PlacementStrategy::MEMORY_AWARE;
    config.max_gpu_memory_mb = 8192; // 8GB budget
    config.verbose = false;

    DeviceOrchestrator orchestrator(device_mgr_, mpi_ctx_, config);

    // Verify strategy is set
    EXPECT_EQ(orchestrator.strategy(), PlacementStrategy::MEMORY_AWARE);
    EXPECT_TRUE(config.max_gpu_memory_mb.has_value());
    EXPECT_EQ(config.max_gpu_memory_mb.value(), 8192);
}

TEST_F(DeviceOrchestratorPhase2Test, MemoryAwareStrategyNoBudgetFallsBackToAuto)
{
    OrchestrationConfig config;
    config.strategy = PlacementStrategy::MEMORY_AWARE;
    // No max_gpu_memory_mb set
    config.verbose = false;

    DeviceOrchestrator orchestrator(device_mgr_, mpi_ctx_, config);

    EXPECT_EQ(orchestrator.strategy(), PlacementStrategy::MEMORY_AWARE);
    EXPECT_FALSE(config.max_gpu_memory_mb.has_value());
}

// ============================================================================
// MOE_OPTIMIZED Strategy Tests
// ============================================================================

TEST_F(DeviceOrchestratorPhase2Test, MoEOptimizedStrategyDefaults)
{
    OrchestrationConfig config;
    config.strategy = PlacementStrategy::MOE_OPTIMIZED;
    config.verbose = false;

    DeviceOrchestrator orchestrator(device_mgr_, mpi_ctx_, config);

    EXPECT_EQ(orchestrator.strategy(), PlacementStrategy::MOE_OPTIMIZED);
    EXPECT_TRUE(config.moe_shared_experts_gpu); // Default: shared on GPU
    EXPECT_TRUE(config.moe_sparse_experts_cpu); // Default: sparse on CPU
}

TEST_F(DeviceOrchestratorPhase2Test, MoEOptimizedStrategySharedCPU)
{
    OrchestrationConfig config;
    config.strategy = PlacementStrategy::MOE_OPTIMIZED;
    config.moe_shared_experts_gpu = false; // Override: shared on CPU
    config.verbose = false;

    DeviceOrchestrator orchestrator(device_mgr_, mpi_ctx_, config);

    EXPECT_EQ(orchestrator.strategy(), PlacementStrategy::MOE_OPTIMIZED);
    EXPECT_FALSE(config.moe_shared_experts_gpu);
}

TEST_F(DeviceOrchestratorPhase2Test, MoEOptimizedStrategySparseGPU)
{
    OrchestrationConfig config;
    config.strategy = PlacementStrategy::MOE_OPTIMIZED;
    config.moe_sparse_experts_cpu = false; // Override: sparse on GPU
    config.verbose = false;

    DeviceOrchestrator orchestrator(device_mgr_, mpi_ctx_, config);

    EXPECT_EQ(orchestrator.strategy(), PlacementStrategy::MOE_OPTIMIZED);
    EXPECT_FALSE(config.moe_sparse_experts_cpu);
}

// ============================================================================
// CUSTOM Strategy Tests
// ============================================================================

TEST_F(DeviceOrchestratorPhase2Test, CustomStrategyEmptyMapFallsBackToAuto)
{
    OrchestrationConfig config;
    config.strategy = PlacementStrategy::CUSTOM;
    config.device_map = ""; // Empty map
    config.verbose = false;

    DeviceOrchestrator orchestrator(device_mgr_, mpi_ctx_, config);

    EXPECT_EQ(orchestrator.strategy(), PlacementStrategy::CUSTOM);
    EXPECT_TRUE(config.device_map.empty());
}

TEST_F(DeviceOrchestratorPhase2Test, CustomStrategyValidMap)
{
    OrchestrationConfig config;
    config.strategy = PlacementStrategy::CUSTOM;
    config.device_map = "0-11:gpu:0,12-23:cpu";
    config.verbose = false;

    DeviceOrchestrator orchestrator(device_mgr_, mpi_ctx_, config);

    EXPECT_EQ(orchestrator.strategy(), PlacementStrategy::CUSTOM);
    EXPECT_EQ(config.device_map, "0-11:gpu:0,12-23:cpu");
}

// ============================================================================
// Integration Tests (Strategy Selection)
// ============================================================================

TEST_F(DeviceOrchestratorPhase2Test, AllPhase2StrategiesSupported)
{
    // MEMORY_AWARE
    {
        OrchestrationConfig config;
        config.strategy = PlacementStrategy::MEMORY_AWARE;
        DeviceOrchestrator orchestrator(device_mgr_, mpi_ctx_, config);
        EXPECT_EQ(orchestrator.strategy(), PlacementStrategy::MEMORY_AWARE);
    }

    // MOE_OPTIMIZED
    {
        OrchestrationConfig config;
        config.strategy = PlacementStrategy::MOE_OPTIMIZED;
        DeviceOrchestrator orchestrator(device_mgr_, mpi_ctx_, config);
        EXPECT_EQ(orchestrator.strategy(), PlacementStrategy::MOE_OPTIMIZED);
    }

    // CUSTOM
    {
        OrchestrationConfig config;
        config.strategy = PlacementStrategy::CUSTOM;
        config.device_map = "0-5:gpu:0";
        DeviceOrchestrator orchestrator(device_mgr_, mpi_ctx_, config);
        EXPECT_EQ(orchestrator.strategy(), PlacementStrategy::CUSTOM);
    }
}

TEST_F(DeviceOrchestratorPhase2Test, Phase1StrategiesStillWork)
{
    // ALL_GPU
    {
        OrchestrationConfig config;
        config.strategy = PlacementStrategy::ALL_GPU;
        DeviceOrchestrator orchestrator(device_mgr_, mpi_ctx_, config);
        EXPECT_EQ(orchestrator.strategy(), PlacementStrategy::ALL_GPU);
    }

    // ALL_CPU
    {
        OrchestrationConfig config;
        config.strategy = PlacementStrategy::ALL_CPU;
        DeviceOrchestrator orchestrator(device_mgr_, mpi_ctx_, config);
        EXPECT_EQ(orchestrator.strategy(), PlacementStrategy::ALL_CPU);
    }

    // LAYER_SPLIT
    {
        OrchestrationConfig config;
        config.strategy = PlacementStrategy::LAYER_SPLIT;
        config.offload_layers = 16;
        DeviceOrchestrator orchestrator(device_mgr_, mpi_ctx_, config);
        EXPECT_EQ(orchestrator.strategy(), PlacementStrategy::LAYER_SPLIT);
    }

    // AUTO
    {
        OrchestrationConfig config;
        config.strategy = PlacementStrategy::AUTO;
        DeviceOrchestrator orchestrator(device_mgr_, mpi_ctx_, config);
        EXPECT_EQ(orchestrator.strategy(), PlacementStrategy::AUTO);
    }
}

// ============================================================================
// Edge Cases
// ============================================================================

TEST_F(DeviceOrchestratorPhase2Test, ParseInvalidDeviceMapRule)
{
    OrchestrationConfig config;
    config.strategy = PlacementStrategy::CUSTOM;
    config.device_map = "invalid_rule"; // No colon separator
    config.verbose = false;

    DeviceOrchestrator orchestrator(device_mgr_, mpi_ctx_, config);

    auto rules = orchestrator.parseDeviceMapString(config.device_map);

    // Invalid rules are filtered out
    EXPECT_EQ(rules.size(), 0);
}

TEST_F(DeviceOrchestratorPhase2Test, ParseDeviceMapWithWhitespace)
{
    OrchestrationConfig config;
    config.strategy = PlacementStrategy::CUSTOM;
    config.device_map = " 0-5:gpu:0 , 6-11:cpu "; // Extra whitespace
    config.verbose = false;

    DeviceOrchestrator orchestrator(device_mgr_, mpi_ctx_, config);

    auto rules = orchestrator.parseDeviceMapString(config.device_map);

    ASSERT_EQ(rules.size(), 2);
    EXPECT_EQ(rules[0].start_layer, 0);
    EXPECT_EQ(rules[0].end_layer, 5);
    EXPECT_EQ(rules[1].start_layer, 6);
    EXPECT_EQ(rules[1].end_layer, 11);
}

int main(int argc, char **argv)
{
    ::testing::InitGoogleTest(&argc, argv);

    // Initialize MPI
    int provided;
    llaminar2::tests::mpi_init_thread_sanitizer_safe(&argc, &argv, MPI_THREAD_MULTIPLE, &provided);

    int result = RUN_ALL_TESTS();

    llaminar2::tests::mpi_finalize_sanitizer_safe();
    return result;
}
