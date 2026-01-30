/**
 * @file Test__LayerDevicePlacement.cpp
 * @brief Unit tests for LayerDevicePlacement and subclasses
 *
 * Tests:
 * - SingleDevicePlacement: all layers on one device
 * - HybridCpuGpuPlacement: CPU spillover with configurable arrangement
 * - PipelineParallelPlacement: layer range ownership
 * - LocalTPPlacement: multi-device TP with weighted distribution
 * - Factory method: fromExecutionPlan()
 *
 * @author David Sanftenberg
 * @date January 2026
 */

#include <gtest/gtest.h>
#include "execution/local_execution/device/LayerDevicePlacement.h"
#include "backends/GlobalDeviceAddress.h"

using namespace llaminar2;

// ============================================================================
// Test Constants
// ============================================================================

constexpr int TEST_TOTAL_LAYERS = 32;
constexpr int TEST_TOTAL_HEADS = 28;
constexpr int TEST_TOTAL_KV_HEADS = 4;

// ============================================================================
// SingleDevicePlacement Tests
// ============================================================================

class Test__SingleDevicePlacement : public ::testing::Test
{
protected:
    GlobalDeviceAddress cuda_device = GlobalDeviceAddress::cuda(0);
    std::unique_ptr<SingleDevicePlacement> placement;

    void SetUp() override
    {
        placement = std::make_unique<SingleDevicePlacement>(
            cuda_device,
            TEST_TOTAL_LAYERS,
            TEST_TOTAL_HEADS,
            TEST_TOTAL_KV_HEADS);
    }
};

TEST_F(Test__SingleDevicePlacement, AllLayersOnSameDevice)
{
    for (int i = 0; i < TEST_TOTAL_LAYERS; ++i)
    {
        EXPECT_EQ(placement->deviceForLayer(i), cuda_device)
            << "Layer " << i << " should be on cuda:0";
    }
}

TEST_F(Test__SingleDevicePlacement, AllStagesOnSameDevice)
{
    EXPECT_EQ(placement->deviceForStage("embedding"), cuda_device);
    EXPECT_EQ(placement->deviceForStage("final_norm"), cuda_device);
    EXPECT_EQ(placement->deviceForStage("lm_head"), cuda_device);
}

TEST_F(Test__SingleDevicePlacement, AllHeadsOnDevice)
{
    EXPECT_EQ(placement->headsForDevice(cuda_device), TEST_TOTAL_HEADS);
    EXPECT_EQ(placement->kvHeadsForDevice(cuda_device), TEST_TOTAL_KV_HEADS);
}

TEST_F(Test__SingleDevicePlacement, ShouldBuildAllLayers)
{
    for (int i = 0; i < TEST_TOTAL_LAYERS; ++i)
    {
        EXPECT_TRUE(placement->shouldBuildLayer(i))
            << "Should build layer " << i;
    }

    // Negative and out-of-range should return false
    EXPECT_FALSE(placement->shouldBuildLayer(-1));
    EXPECT_FALSE(placement->shouldBuildLayer(TEST_TOTAL_LAYERS));
}

TEST_F(Test__SingleDevicePlacement, NoSharding)
{
    EXPECT_EQ(placement->totalShards(), 1);
    EXPECT_EQ(placement->shardIndexForDevice(cuda_device), 0);
}

TEST_F(Test__SingleDevicePlacement, SingleDeviceInList)
{
    auto devices = placement->allDevices();
    ASSERT_EQ(devices.size(), 1);
    EXPECT_EQ(devices[0], cuda_device);
}

TEST_F(Test__SingleDevicePlacement, PrimaryDevice)
{
    EXPECT_EQ(placement->primaryDevice(), cuda_device);
}

// ============================================================================
// HybridCpuGpuPlacement Tests
// ============================================================================

class Test__HybridCpuGpuPlacement : public ::testing::Test
{
protected:
    GlobalDeviceAddress gpu_device = GlobalDeviceAddress::cuda(0);
    GlobalDeviceAddress cpu_device = GlobalDeviceAddress::cpu();
};

TEST_F(Test__HybridCpuGpuPlacement, GpuFirst_GpuLayers)
{
    // GPU first: layers 0-19 on GPU, 20-31 on CPU
    constexpr int GPU_LAYERS = 20;
    HybridCpuGpuPlacement placement(
        gpu_device, cpu_device, GPU_LAYERS, /*cpu_first=*/false,
        TEST_TOTAL_LAYERS, TEST_TOTAL_HEADS, TEST_TOTAL_KV_HEADS);

    // Check GPU layers
    for (int i = 0; i < GPU_LAYERS; ++i)
    {
        EXPECT_EQ(placement.deviceForLayer(i), gpu_device)
            << "Layer " << i << " should be on GPU";
        EXPECT_TRUE(placement.isGpuLayer(i));
    }

    // Check CPU layers
    for (int i = GPU_LAYERS; i < TEST_TOTAL_LAYERS; ++i)
    {
        EXPECT_EQ(placement.deviceForLayer(i), cpu_device)
            << "Layer " << i << " should be on CPU";
        EXPECT_FALSE(placement.isGpuLayer(i));
    }
}

TEST_F(Test__HybridCpuGpuPlacement, CpuFirst_CpuLayers)
{
    // CPU first: layers 0-11 on CPU, 12-31 on GPU
    constexpr int GPU_LAYERS = 20;
    HybridCpuGpuPlacement placement(
        gpu_device, cpu_device, GPU_LAYERS, /*cpu_first=*/true,
        TEST_TOTAL_LAYERS, TEST_TOTAL_HEADS, TEST_TOTAL_KV_HEADS);

    int cpu_layers = TEST_TOTAL_LAYERS - GPU_LAYERS; // 12

    // Check CPU layers (first 12)
    for (int i = 0; i < cpu_layers; ++i)
    {
        EXPECT_EQ(placement.deviceForLayer(i), cpu_device)
            << "Layer " << i << " should be on CPU";
        EXPECT_FALSE(placement.isGpuLayer(i));
    }

    // Check GPU layers (last 20)
    for (int i = cpu_layers; i < TEST_TOTAL_LAYERS; ++i)
    {
        EXPECT_EQ(placement.deviceForLayer(i), gpu_device)
            << "Layer " << i << " should be on GPU";
        EXPECT_TRUE(placement.isGpuLayer(i));
    }
}

TEST_F(Test__HybridCpuGpuPlacement, StageDevices_GpuFirst)
{
    HybridCpuGpuPlacement placement(
        gpu_device, cpu_device, 20, /*cpu_first=*/false,
        TEST_TOTAL_LAYERS, TEST_TOTAL_HEADS, TEST_TOTAL_KV_HEADS);

    // Embedding near first layers (GPU when gpu_first)
    EXPECT_EQ(placement.deviceForStage("embedding"), gpu_device);
    // LM head near last layers (CPU when gpu_first)
    EXPECT_EQ(placement.deviceForStage("lm_head"), cpu_device);
}

TEST_F(Test__HybridCpuGpuPlacement, StageDevices_CpuFirst)
{
    HybridCpuGpuPlacement placement(
        gpu_device, cpu_device, 20, /*cpu_first=*/true,
        TEST_TOTAL_LAYERS, TEST_TOTAL_HEADS, TEST_TOTAL_KV_HEADS);

    // Embedding near first layers (CPU when cpu_first)
    EXPECT_EQ(placement.deviceForStage("embedding"), cpu_device);
    // LM head near last layers (GPU when cpu_first)
    EXPECT_EQ(placement.deviceForStage("lm_head"), gpu_device);
}

TEST_F(Test__HybridCpuGpuPlacement, AllLayersOnGpu)
{
    HybridCpuGpuPlacement placement(
        gpu_device, cpu_device, TEST_TOTAL_LAYERS, /*cpu_first=*/false,
        TEST_TOTAL_LAYERS, TEST_TOTAL_HEADS, TEST_TOTAL_KV_HEADS);

    for (int i = 0; i < TEST_TOTAL_LAYERS; ++i)
    {
        EXPECT_EQ(placement.deviceForLayer(i), gpu_device);
    }

    EXPECT_EQ(placement.gpuLayerCount(), TEST_TOTAL_LAYERS);
    EXPECT_EQ(placement.cpuLayerCount(), 0);
}

TEST_F(Test__HybridCpuGpuPlacement, AllLayersOnCpu)
{
    HybridCpuGpuPlacement placement(
        gpu_device, cpu_device, 0, /*cpu_first=*/false,
        TEST_TOTAL_LAYERS, TEST_TOTAL_HEADS, TEST_TOTAL_KV_HEADS);

    for (int i = 0; i < TEST_TOTAL_LAYERS; ++i)
    {
        EXPECT_EQ(placement.deviceForLayer(i), cpu_device);
    }

    EXPECT_EQ(placement.gpuLayerCount(), 0);
    EXPECT_EQ(placement.cpuLayerCount(), TEST_TOTAL_LAYERS);
}

TEST_F(Test__HybridCpuGpuPlacement, NoSharding)
{
    HybridCpuGpuPlacement placement(
        gpu_device, cpu_device, 20, /*cpu_first=*/false,
        TEST_TOTAL_LAYERS, TEST_TOTAL_HEADS, TEST_TOTAL_KV_HEADS);

    EXPECT_EQ(placement.totalShards(), 1);
    EXPECT_EQ(placement.headsForDevice(gpu_device), TEST_TOTAL_HEADS);
    EXPECT_EQ(placement.headsForDevice(cpu_device), TEST_TOTAL_HEADS);
}

TEST_F(Test__HybridCpuGpuPlacement, InvalidGpuLayers_Throws)
{
    EXPECT_THROW({ HybridCpuGpuPlacement placement(
                       gpu_device, cpu_device, -1, false,
                       TEST_TOTAL_LAYERS, TEST_TOTAL_HEADS, TEST_TOTAL_KV_HEADS); }, std::invalid_argument);

    EXPECT_THROW({ HybridCpuGpuPlacement placement(
                       gpu_device, cpu_device, TEST_TOTAL_LAYERS + 1, false,
                       TEST_TOTAL_LAYERS, TEST_TOTAL_HEADS, TEST_TOTAL_KV_HEADS); }, std::invalid_argument);
}

// ============================================================================
// PipelineParallelPlacement Tests
// ============================================================================

class Test__PipelineParallelPlacement : public ::testing::Test
{
protected:
    GlobalDeviceAddress device = GlobalDeviceAddress::cuda(0);
};

TEST_F(Test__PipelineParallelPlacement, FirstStage_HasEmbedding)
{
    // First stage: layers 0-15, has embedding, no LM head
    PipelineParallelPlacement placement(
        device,
        /*first_layer=*/0, /*last_layer=*/15,
        TEST_TOTAL_LAYERS, TEST_TOTAL_HEADS, TEST_TOTAL_KV_HEADS,
        /*shard_index=*/0, /*total_shards=*/1,
        /*has_embedding=*/true, /*has_lm_head=*/false);

    EXPECT_TRUE(placement.shouldBuildEmbedding());
    EXPECT_FALSE(placement.shouldBuildLMHead(TEST_TOTAL_LAYERS));
}

TEST_F(Test__PipelineParallelPlacement, LastStage_HasLMHead)
{
    // Last stage: layers 16-31, no embedding, has LM head
    PipelineParallelPlacement placement(
        device,
        /*first_layer=*/16, /*last_layer=*/31,
        TEST_TOTAL_LAYERS, TEST_TOTAL_HEADS, TEST_TOTAL_KV_HEADS,
        /*shard_index=*/0, /*total_shards=*/1,
        /*has_embedding=*/false, /*has_lm_head=*/true);

    EXPECT_FALSE(placement.shouldBuildEmbedding());
    EXPECT_TRUE(placement.shouldBuildLMHead(TEST_TOTAL_LAYERS));
}

TEST_F(Test__PipelineParallelPlacement, MiddleStage_NoEmbeddingOrLMHead)
{
    // Middle stage: layers 8-15, no embedding, no LM head
    PipelineParallelPlacement placement(
        device,
        /*first_layer=*/8, /*last_layer=*/15,
        TEST_TOTAL_LAYERS, TEST_TOTAL_HEADS, TEST_TOTAL_KV_HEADS,
        /*shard_index=*/0, /*total_shards=*/1,
        /*has_embedding=*/false, /*has_lm_head=*/false);

    EXPECT_FALSE(placement.shouldBuildEmbedding());
    EXPECT_FALSE(placement.shouldBuildLMHead(TEST_TOTAL_LAYERS));
}

TEST_F(Test__PipelineParallelPlacement, ShouldBuildLayer_InRange)
{
    PipelineParallelPlacement placement(
        device,
        /*first_layer=*/8, /*last_layer=*/15,
        TEST_TOTAL_LAYERS, TEST_TOTAL_HEADS, TEST_TOTAL_KV_HEADS,
        /*shard_index=*/0, /*total_shards=*/1,
        /*has_embedding=*/false, /*has_lm_head=*/false);

    // Before range
    for (int i = 0; i < 8; ++i)
    {
        EXPECT_FALSE(placement.shouldBuildLayer(i))
            << "Should not build layer " << i;
    }

    // In range
    for (int i = 8; i <= 15; ++i)
    {
        EXPECT_TRUE(placement.shouldBuildLayer(i))
            << "Should build layer " << i;
    }

    // After range
    for (int i = 16; i < TEST_TOTAL_LAYERS; ++i)
    {
        EXPECT_FALSE(placement.shouldBuildLayer(i))
            << "Should not build layer " << i;
    }
}

TEST_F(Test__PipelineParallelPlacement, LayerCount)
{
    PipelineParallelPlacement placement(
        device,
        /*first_layer=*/8, /*last_layer=*/15,
        TEST_TOTAL_LAYERS, TEST_TOTAL_HEADS, TEST_TOTAL_KV_HEADS,
        /*shard_index=*/0, /*total_shards=*/1,
        /*has_embedding=*/false, /*has_lm_head=*/false);

    EXPECT_EQ(placement.firstLayer(), 8);
    EXPECT_EQ(placement.lastLayer(), 15);
    EXPECT_EQ(placement.layerCount(), 8);
}

TEST_F(Test__PipelineParallelPlacement, WithTPSharding)
{
    // PP + TP: layer range with 2-way sharding
    constexpr int LOCAL_HEADS = 14;   // Half of 28
    constexpr int LOCAL_KV_HEADS = 2; // Half of 4

    PipelineParallelPlacement placement(
        device,
        /*first_layer=*/0, /*last_layer=*/15,
        TEST_TOTAL_LAYERS, LOCAL_HEADS, LOCAL_KV_HEADS,
        /*shard_index=*/0, /*total_shards=*/2,
        /*has_embedding=*/true, /*has_lm_head=*/false);

    EXPECT_EQ(placement.totalShards(), 2);
    EXPECT_EQ(placement.shardIndexForDevice(device), 0);
    EXPECT_EQ(placement.headsForDevice(device), LOCAL_HEADS);
    EXPECT_EQ(placement.kvHeadsForDevice(device), LOCAL_KV_HEADS);
}

// ============================================================================
// LocalTPPlacement Tests
// ============================================================================

class Test__LocalTPPlacement : public ::testing::Test
{
protected:
    GlobalDeviceAddress cuda0 = GlobalDeviceAddress::cuda(0);
    GlobalDeviceAddress cuda1 = GlobalDeviceAddress::cuda(1);
    GlobalDeviceAddress rocm0 = GlobalDeviceAddress::rocm(0);
};

TEST_F(Test__LocalTPPlacement, TwoDevices_EqualWeights)
{
    LocalTPPlacement placement(
        {cuda0, cuda1},
        {}, // Empty weights = equal distribution
        TEST_TOTAL_LAYERS, TEST_TOTAL_HEADS, TEST_TOTAL_KV_HEADS);

    EXPECT_EQ(placement.totalShards(), 2);
    EXPECT_EQ(placement.shardIndexForDevice(cuda0), 0);
    EXPECT_EQ(placement.shardIndexForDevice(cuda1), 1);

    // Equal split: 28/2 = 14 heads each
    EXPECT_EQ(placement.headsForDevice(cuda0), 14);
    EXPECT_EQ(placement.headsForDevice(cuda1), 14);

    // KV heads: 4/2 = 2 each
    EXPECT_EQ(placement.kvHeadsForDevice(cuda0), 2);
    EXPECT_EQ(placement.kvHeadsForDevice(cuda1), 2);
}

TEST_F(Test__LocalTPPlacement, TwoDevices_WeightedDistribution)
{
    // NVIDIA 73%, AMD 27%
    LocalTPPlacement placement(
        {cuda0, rocm0},
        {0.73f, 0.27f},
        TEST_TOTAL_LAYERS, TEST_TOTAL_HEADS, TEST_TOTAL_KV_HEADS);

    // 28 * 0.73 = 20.44 -> 20 heads for CUDA
    // 28 - 20 = 8 heads for ROCm (last device gets remainder)
    int cuda_heads = placement.headsForDevice(cuda0);
    int rocm_heads = placement.headsForDevice(rocm0);

    EXPECT_EQ(cuda_heads + rocm_heads, TEST_TOTAL_HEADS);
    EXPECT_GT(cuda_heads, rocm_heads);
    EXPECT_NEAR(static_cast<float>(cuda_heads) / TEST_TOTAL_HEADS, 0.73f, 0.1f);
}

TEST_F(Test__LocalTPPlacement, ThreeDevices_EqualWeights)
{
    GlobalDeviceAddress cuda2 = GlobalDeviceAddress::cuda(2);

    LocalTPPlacement placement(
        {cuda0, cuda1, cuda2},
        {}, // Equal distribution
        TEST_TOTAL_LAYERS, TEST_TOTAL_HEADS, TEST_TOTAL_KV_HEADS);

    EXPECT_EQ(placement.totalShards(), 3);

    // 28 heads / 3 = 9, 9, 10 (last gets remainder)
    int h0 = placement.headsForDevice(cuda0);
    int h1 = placement.headsForDevice(cuda1);
    int h2 = placement.headsForDevice(cuda2);

    EXPECT_EQ(h0 + h1 + h2, TEST_TOTAL_HEADS);
}

TEST_F(Test__LocalTPPlacement, AllLayersShouldBuild)
{
    LocalTPPlacement placement(
        {cuda0, cuda1},
        {},
        TEST_TOTAL_LAYERS, TEST_TOTAL_HEADS, TEST_TOTAL_KV_HEADS);

    // Local TP: this rank builds all layers
    for (int i = 0; i < TEST_TOTAL_LAYERS; ++i)
    {
        EXPECT_TRUE(placement.shouldBuildLayer(i));
    }
}

TEST_F(Test__LocalTPPlacement, PrimaryDevice)
{
    LocalTPPlacement placement(
        {cuda0, cuda1},
        {},
        TEST_TOTAL_LAYERS, TEST_TOTAL_HEADS, TEST_TOTAL_KV_HEADS);

    EXPECT_EQ(placement.primaryDevice(), cuda0);
}

TEST_F(Test__LocalTPPlacement, AllDevicesReturned)
{
    LocalTPPlacement placement(
        {cuda0, cuda1, rocm0},
        {},
        TEST_TOTAL_LAYERS, TEST_TOTAL_HEADS, TEST_TOTAL_KV_HEADS);

    auto devices = placement.allDevices();
    ASSERT_EQ(devices.size(), 3);
    EXPECT_EQ(devices[0], cuda0);
    EXPECT_EQ(devices[1], cuda1);
    EXPECT_EQ(devices[2], rocm0);
}

TEST_F(Test__LocalTPPlacement, EmptyDevices_Throws)
{
    EXPECT_THROW({ LocalTPPlacement placement(
                       {}, // Empty!
                       {},
                       TEST_TOTAL_LAYERS, TEST_TOTAL_HEADS, TEST_TOTAL_KV_HEADS); }, std::invalid_argument);
}

TEST_F(Test__LocalTPPlacement, MismatchedWeights_Throws)
{
    EXPECT_THROW({ LocalTPPlacement placement(
                       {cuda0, cuda1},
                       {0.5f}, // Only one weight for two devices
                       TEST_TOTAL_LAYERS, TEST_TOTAL_HEADS, TEST_TOTAL_KV_HEADS); }, std::invalid_argument);
}

TEST_F(Test__LocalTPPlacement, UnknownDevice_ReturnsAll)
{
    LocalTPPlacement placement(
        {cuda0, cuda1},
        {},
        TEST_TOTAL_LAYERS, TEST_TOTAL_HEADS, TEST_TOTAL_KV_HEADS);

    GlobalDeviceAddress unknown = GlobalDeviceAddress::cuda(99);
    EXPECT_EQ(placement.headsForDevice(unknown), TEST_TOTAL_HEADS);
}

// ============================================================================
// Factory Method Tests
// ============================================================================

class Test__LayerDevicePlacement_Factory : public ::testing::Test
{
protected:
    RankExecutionPlan plan;

    void SetUp() override
    {
        plan = RankExecutionPlan{};
        plan.rank = 0;
        plan.hostname = "localhost";
        plan.numa_node = 0;
        plan.primary_device = GlobalDeviceAddress::cuda(0);
        plan.first_layer = 0;
        plan.last_layer = TEST_TOTAL_LAYERS - 1;
        plan.has_embedding = true;
        plan.has_lm_head = true;
    }
};

TEST_F(Test__LayerDevicePlacement_Factory, SingleDevice_NoTP_NoPP)
{
    auto placement = LayerDevicePlacement::fromExecutionPlan(
        plan, TEST_TOTAL_LAYERS, TEST_TOTAL_HEADS, TEST_TOTAL_KV_HEADS);

    ASSERT_NE(placement, nullptr);
    EXPECT_EQ(placement->totalShards(), 1);
    EXPECT_TRUE(placement->shouldBuildLayer(0));
    EXPECT_TRUE(placement->shouldBuildLayer(TEST_TOTAL_LAYERS - 1));
}

TEST_F(Test__LayerDevicePlacement_Factory, PipelineParallel_FirstStage)
{
    plan.first_layer = 0;
    plan.last_layer = 15;
    plan.has_embedding = true;
    plan.has_lm_head = false;
    plan.next_rank = 1; // Has next rank -> PP enabled

    auto placement = LayerDevicePlacement::fromExecutionPlan(
        plan, TEST_TOTAL_LAYERS, TEST_TOTAL_HEADS, TEST_TOTAL_KV_HEADS);

    ASSERT_NE(placement, nullptr);
    EXPECT_TRUE(placement->shouldBuildLayer(0));
    EXPECT_TRUE(placement->shouldBuildLayer(15));
    EXPECT_FALSE(placement->shouldBuildLayer(16));
    EXPECT_TRUE(placement->shouldBuildEmbedding());
    EXPECT_FALSE(placement->shouldBuildLMHead(TEST_TOTAL_LAYERS));
}

TEST_F(Test__LayerDevicePlacement_Factory, PipelineParallel_LastStage)
{
    plan.first_layer = 16;
    plan.last_layer = 31;
    plan.has_embedding = false;
    plan.has_lm_head = true;
    plan.prev_rank = 0; // Has prev rank -> PP enabled

    auto placement = LayerDevicePlacement::fromExecutionPlan(
        plan, TEST_TOTAL_LAYERS, TEST_TOTAL_HEADS, TEST_TOTAL_KV_HEADS);

    ASSERT_NE(placement, nullptr);
    EXPECT_FALSE(placement->shouldBuildLayer(0));
    EXPECT_FALSE(placement->shouldBuildLayer(15));
    EXPECT_TRUE(placement->shouldBuildLayer(16));
    EXPECT_TRUE(placement->shouldBuildLayer(31));
    EXPECT_FALSE(placement->shouldBuildEmbedding());
    EXPECT_TRUE(placement->shouldBuildLMHead(TEST_TOTAL_LAYERS));
}

TEST_F(Test__LayerDevicePlacement_Factory, LocalTP_TwoDevices)
{
    plan.local_tp_devices = {
        GlobalDeviceAddress::cuda(0),
        GlobalDeviceAddress::cuda(1)};
    plan.local_tp_weights = {0.5f, 0.5f};

    auto placement = LayerDevicePlacement::fromExecutionPlan(
        plan, TEST_TOTAL_LAYERS, TEST_TOTAL_HEADS, TEST_TOTAL_KV_HEADS);

    ASSERT_NE(placement, nullptr);
    EXPECT_EQ(placement->totalShards(), 2);
    EXPECT_TRUE(placement->shouldBuildLayer(0));
    EXPECT_TRUE(placement->shouldBuildLayer(TEST_TOTAL_LAYERS - 1));
}

TEST_F(Test__LayerDevicePlacement_Factory, LocalTP_ProportionalWeights)
{
    // NVIDIA 73%, AMD 27%
    plan.local_tp_devices = {
        GlobalDeviceAddress::cuda(0),
        GlobalDeviceAddress::rocm(0)};
    plan.local_tp_weights = {0.73f, 0.27f};

    auto placement = LayerDevicePlacement::fromExecutionPlan(
        plan, TEST_TOTAL_LAYERS, TEST_TOTAL_HEADS, TEST_TOTAL_KV_HEADS);

    ASSERT_NE(placement, nullptr);

    int cuda_heads = placement->headsForDevice(GlobalDeviceAddress::cuda(0));
    int rocm_heads = placement->headsForDevice(GlobalDeviceAddress::rocm(0));

    EXPECT_EQ(cuda_heads + rocm_heads, TEST_TOTAL_HEADS);
    EXPECT_GT(cuda_heads, rocm_heads);
}

// ============================================================================
// Edge Cases
// ============================================================================

TEST(Test__LayerDevicePlacement_EdgeCases, SingleLayer)
{
    SingleDevicePlacement placement(
        GlobalDeviceAddress::cuda(0),
        1, // Single layer
        4, // Few heads
        1);

    EXPECT_TRUE(placement.shouldBuildLayer(0));
    EXPECT_FALSE(placement.shouldBuildLayer(1));
    EXPECT_EQ(placement.headsForDevice(GlobalDeviceAddress::cuda(0)), 4);
}

TEST(Test__LayerDevicePlacement_EdgeCases, PP_SingleLayerPerStage)
{
    PipelineParallelPlacement placement(
        GlobalDeviceAddress::cuda(0),
        5, 5, // Single layer range
        TEST_TOTAL_LAYERS, TEST_TOTAL_HEADS, TEST_TOTAL_KV_HEADS,
        0, 1, false, false);

    EXPECT_EQ(placement.layerCount(), 1);
    EXPECT_FALSE(placement.shouldBuildLayer(4));
    EXPECT_TRUE(placement.shouldBuildLayer(5));
    EXPECT_FALSE(placement.shouldBuildLayer(6));
}
