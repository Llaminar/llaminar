/**
 * @file Test__DeviceOrchestrator.cpp
 * @brief Tests for DeviceOrchestrator
 * @author David Sanftenberg
 * @date 2025-10-24
 */

#include <gtest/gtest.h>
#include "loaders/DeviceOrchestrator.h"
#include "loaders/ModelContext.h"
#include "backends/ComputeBackend.h"
#include "utils/MPIContext.h"
#include "TestMPIUtils.h"

using namespace llaminar2;

class Test__DeviceOrchestrator : public ::testing::Test
{
protected:
    void SetUp() override
    {
        // Initialize device manager
        auto &dm = DeviceManager::instance();
        dm.initialize();
        device_mgr_ = std::shared_ptr<DeviceManager>(&dm, [](DeviceManager *) {});

        // Create MPI context (single rank for unit tests)
        mpi_ctx_ = MPIContextFactory::global();
    }

    std::shared_ptr<DeviceManager> device_mgr_;
    std::shared_ptr<MPIContext> mpi_ctx_;
};

/**
 * Test: ALL_GPU strategy
 */
TEST_F(Test__DeviceOrchestrator, AllGPUStrategy)
{
    OrchestrationConfig config;
    config.strategy = WeightPlacementStrategy::ALL_GPU;
    config.gpu_device_idx = 0;
    config.verbose = true;

    auto orchestrator = std::make_shared<DeviceOrchestrator>(
        device_mgr_, mpi_ctx_, config);

    EXPECT_EQ(WeightPlacementStrategy::ALL_GPU, orchestrator->strategy());

    // Create mock model context (for testing)
    auto model_ctx = ModelContext::createForTesting("test.gguf");

    auto placement_map = orchestrator->createPlacementMap(model_ctx);
    ASSERT_NE(nullptr, placement_map);

    // All weights should go to GPU device 0
    EXPECT_EQ(0, placement_map->defaultDevice());
    EXPECT_EQ(0, placement_map->getDeviceForWeight("token_embd.weight"));
    EXPECT_EQ(0, placement_map->getDeviceForWeight("blk.0.attn_q.weight"));
    EXPECT_EQ(0, placement_map->getDeviceForWeight("blk.5.ffn_gate.weight"));
    EXPECT_EQ(0, placement_map->getDeviceForWeight("output.weight"));
}

/**
 * Test: ALL_CPU strategy
 */
TEST_F(Test__DeviceOrchestrator, AllCPUStrategy)
{
    OrchestrationConfig config;
    config.strategy = WeightPlacementStrategy::ALL_CPU;
    config.cpu_device_idx = 0; // Assume CPU is device 0

    auto orchestrator = std::make_shared<DeviceOrchestrator>(
        device_mgr_, mpi_ctx_, config);

    EXPECT_EQ(WeightPlacementStrategy::ALL_CPU, orchestrator->strategy());

    auto model_ctx = ModelContext::createForTesting("test.gguf");
    auto placement_map = orchestrator->createPlacementMap(model_ctx);

    ASSERT_NE(nullptr, placement_map);

    // All weights should go to CPU
    EXPECT_EQ(0, placement_map->defaultDevice());
    EXPECT_EQ(0, placement_map->getDeviceForWeight("token_embd.weight"));
    EXPECT_EQ(0, placement_map->getDeviceForWeight("blk.0.attn_q.weight"));
}

/**
 * Test: LAYER_SPLIT strategy with offload_layers
 */
TEST_F(Test__DeviceOrchestrator, LayerSplitStrategy)
{
    OrchestrationConfig config;
    config.strategy = WeightPlacementStrategy::LAYER_SPLIT;
    config.gpu_device_idx = 1; // GPU
    config.cpu_device_idx = 0; // CPU
    config.offload_layers = 4; // First 4 layers on GPU
    config.verbose = true;

    auto orchestrator = std::make_shared<DeviceOrchestrator>(
        device_mgr_, mpi_ctx_, config);

    // Create model with 12 layers to support testing offload_layers=4 and checking blk.10
    auto model_ctx = ModelContext::createForTesting("test.gguf", nullptr, 12);
    auto placement_map = orchestrator->createPlacementMap(model_ctx);

    ASSERT_NE(nullptr, placement_map);

    // Default should be CPU
    EXPECT_EQ(0, placement_map->defaultDevice());

    // Embeddings on GPU (pattern)
    EXPECT_EQ(1, placement_map->getDeviceForWeight("token_embd.weight"));

    // First 4 layers on GPU
    EXPECT_EQ(1, placement_map->getDeviceForWeight("blk.0.attn_q.weight"));
    EXPECT_EQ(1, placement_map->getDeviceForWeight("blk.1.attn_q.weight"));
    EXPECT_EQ(1, placement_map->getDeviceForWeight("blk.2.attn_q.weight"));
    EXPECT_EQ(1, placement_map->getDeviceForWeight("blk.3.attn_q.weight"));

    // Layer 4+ on CPU
    EXPECT_EQ(0, placement_map->getDeviceForWeight("blk.4.attn_q.weight"));
    EXPECT_EQ(0, placement_map->getDeviceForWeight("blk.10.attn_q.weight"));

    // Output on GPU (pattern)
    EXPECT_EQ(1, placement_map->getDeviceForWeight("output.weight"));
}

/**
 * Test: LAYER_SPLIT with zero offload_layers (all CPU except embeddings/output)
 */
TEST_F(Test__DeviceOrchestrator, LayerSplitZeroOffload)
{
    OrchestrationConfig config;
    config.strategy = WeightPlacementStrategy::LAYER_SPLIT;
    config.gpu_device_idx = 1;
    config.cpu_device_idx = 0;
    config.offload_layers = 0; // No layers on GPU

    auto orchestrator = std::make_shared<DeviceOrchestrator>(
        device_mgr_, mpi_ctx_, config);

    auto model_ctx = ModelContext::createForTesting("test.gguf", nullptr, 12);
    auto placement_map = orchestrator->createPlacementMap(model_ctx);

    // Embeddings still on GPU (for performance)
    EXPECT_EQ(1, placement_map->getDeviceForWeight("token_embd.weight"));

    // All layers on CPU
    EXPECT_EQ(0, placement_map->getDeviceForWeight("blk.0.attn_q.weight"));
    EXPECT_EQ(0, placement_map->getDeviceForWeight("blk.5.attn_q.weight"));

    // Output on GPU
    EXPECT_EQ(1, placement_map->getDeviceForWeight("output.weight"));
}

/**
 * Test: AUTO strategy (should select GPU if available, else CPU)
 */
TEST_F(Test__DeviceOrchestrator, AutoStrategy)
{
    OrchestrationConfig config;
    config.strategy = WeightPlacementStrategy::AUTO;
    config.gpu_device_idx = 0;
    config.verbose = false;

    auto orchestrator = std::make_shared<DeviceOrchestrator>(
        device_mgr_, mpi_ctx_, config);

    EXPECT_EQ(WeightPlacementStrategy::AUTO, orchestrator->strategy());

    auto model_ctx = ModelContext::createForTesting("test.gguf");
    auto placement_map = orchestrator->createPlacementMap(model_ctx);

    ASSERT_NE(nullptr, placement_map);

    // AUTO should have chosen a strategy
    // (Result depends on whether GPU is available, but should always succeed)
    int default_device = placement_map->defaultDevice();
    EXPECT_GE(default_device, 0);
}

/**
 * Test: Configuration accessor
 */
TEST_F(Test__DeviceOrchestrator, ConfigAccessor)
{
    OrchestrationConfig config;
    config.strategy = WeightPlacementStrategy::LAYER_SPLIT;
    config.gpu_device_idx = 1;
    config.offload_layers = 8;
    config.verbose = true;

    auto orchestrator = std::make_shared<DeviceOrchestrator>(
        device_mgr_, mpi_ctx_, config);

    const auto &retrieved_config = orchestrator->config();
    EXPECT_EQ(WeightPlacementStrategy::LAYER_SPLIT, retrieved_config.strategy);
    EXPECT_EQ(1, retrieved_config.gpu_device_idx);
    EXPECT_EQ(8, retrieved_config.offload_layers);
    EXPECT_TRUE(retrieved_config.verbose);
}

/**
 * Test: CPU device index auto-detection
 */
TEST_F(Test__DeviceOrchestrator, CPUAutoDetection)
{
    OrchestrationConfig config;
    config.strategy = WeightPlacementStrategy::ALL_CPU;
    config.cpu_device_idx = -1; // Auto-detect

    auto orchestrator = std::make_shared<DeviceOrchestrator>(
        device_mgr_, mpi_ctx_, config);

    // Should have detected CPU device
    EXPECT_GE(orchestrator->config().cpu_device_idx, 0);
}

/**
 * Test: Verbose logging control
 */
TEST_F(Test__DeviceOrchestrator, VerboseLogging)
{
    // Test with verbose on
    OrchestrationConfig config_verbose;
    config_verbose.strategy = WeightPlacementStrategy::ALL_GPU;
    config_verbose.verbose = true;

    auto orch_verbose = std::make_shared<DeviceOrchestrator>(
        device_mgr_, mpi_ctx_, config_verbose);

    auto model_ctx = ModelContext::createForTesting("test.gguf");

    // Should log placement decisions (check via stdout capture if needed)
    auto map1 = orch_verbose->createPlacementMap(model_ctx);
    ASSERT_NE(nullptr, map1);

    // Test with verbose off
    OrchestrationConfig config_quiet;
    config_quiet.strategy = WeightPlacementStrategy::ALL_GPU;
    config_quiet.verbose = false;

    auto orch_quiet = std::make_shared<DeviceOrchestrator>(
        device_mgr_, mpi_ctx_, config_quiet);

    auto map2 = orch_quiet->createPlacementMap(model_ctx);
    ASSERT_NE(nullptr, map2);
}

int main(int argc, char **argv)
{
    ::testing::InitGoogleTest(&argc, argv);

    // Initialize MPI for tests
    int provided;
    llaminar2::tests::mpi_init_thread_sanitizer_safe(&argc, &argv, MPI_THREAD_MULTIPLE, &provided);

    int result = RUN_ALL_TESTS();

    llaminar2::tests::mpi_finalize_sanitizer_safe();
    return result;
}
