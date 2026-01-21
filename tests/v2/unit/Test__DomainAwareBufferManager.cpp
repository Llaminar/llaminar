/**
 * @file Test__DomainAwareBufferManager.cpp
 * @brief Unit tests for DomainAwareBufferManager
 *
 * Tests domain-aware buffer allocation based on layer placement configuration.
 * Verifies that GPU layers get GPU buffers and CPU layers get NUMA-local buffers.
 *
 * @author David Sanftenberg
 * @date 2026-01-21
 */

#include <gtest/gtest.h>
#include "execution/DomainAwareBufferManager.h"
#include "execution/BufferRole.h" // For BufferTensorType
#include "config/LayerPlacementConfig.h"
#include "memory/NUMAAllocator.h"
#include "backends/DeviceId.h"
#include "tensors/TensorClasses.h"

using namespace llaminar2;

// =============================================================================
// Construction Tests
// =============================================================================

TEST(Test__DomainAwareBufferManager, ConstructsWithDefaultConfig)
{
    DomainAwareBufferConfig config;
    DomainAwareBufferManager manager(config);

    EXPECT_EQ(manager.bufferCount(), 0);
    EXPECT_EQ(manager.getStats().total_bytes(), 0);
    EXPECT_EQ(manager.getStats().total_buffers(), 0);
}

TEST(Test__DomainAwareBufferManager, ConstructsWithPlacementConfig)
{
    // Create placement config: all layers on GPU
    auto placement = LayerPlacementConfig::allOnDevice(DeviceId::cuda(0), 28);

    DomainAwareBufferConfig config;
    config.placement_config = &placement;

    DomainAwareBufferManager manager(config);

    EXPECT_EQ(manager.bufferCount(), 0);
}

TEST(Test__DomainAwareBufferManager, ConstructsWithNUMAAllocator)
{
    DomainAwareBufferConfig config;
    config.numa_allocator = &NUMAAllocator::instance();
    config.default_numa_node = 0;

    DomainAwareBufferManager manager(config);

    EXPECT_EQ(manager.bufferCount(), 0);
}

// =============================================================================
// allocateForLayer Tests
// =============================================================================

TEST(Test__DomainAwareBufferManager, AllocateForLayerUsesPlacement)
{
    // Create placement config: layers 0-3 on CPU, rest on CUDA:0
    auto placement = LayerPlacementConfig::cpuFirstLayers(4, 28, DeviceId::cuda(0));

    DomainAwareBufferConfig config;
    config.placement_config = &placement;

    DomainAwareBufferManager manager(config);

    // Verify layer placement is used
    EXPECT_TRUE(manager.isCPULayer(0));
    EXPECT_TRUE(manager.isCPULayer(3));
    EXPECT_TRUE(manager.isGPULayer(4));
    EXPECT_TRUE(manager.isGPULayer(27));
}

TEST(Test__DomainAwareBufferManager, AllocateForCPULayerReturnsCPUTensor)
{
    // Create placement config: layers 0-3 on CPU
    auto placement = LayerPlacementConfig::cpuFirstLayers(4, 28, DeviceId::cuda(0));

    DomainAwareBufferConfig config;
    config.placement_config = &placement;

    DomainAwareBufferManager manager(config);

    // Allocate buffer for CPU layer
    auto *tensor = manager.allocateForLayer(0, "attention_output", {128, 896}, BufferTensorType::FP32);

    ASSERT_NE(tensor, nullptr);
    EXPECT_TRUE(tensor->home_device().is_cpu());

    // Verify stats
    EXPECT_EQ(manager.getStats().cpu_buffer_count, 1);
    EXPECT_EQ(manager.getStats().gpu_buffer_count, 0);
    EXPECT_GT(manager.getStats().cpu_bytes_allocated, 0);
}

TEST(Test__DomainAwareBufferManager, AllocateForGPULayerReturnsGPUTensor)
{
    // Create placement config: all layers on GPU
    auto placement = LayerPlacementConfig::allOnDevice(DeviceId::cuda(0), 28);

    DomainAwareBufferConfig config;
    config.placement_config = &placement;

    DomainAwareBufferManager manager(config);

    // Allocate buffer for GPU layer
    auto *tensor = manager.allocateForLayer(5, "attention_output", {128, 896}, BufferTensorType::FP32);

    ASSERT_NE(tensor, nullptr);
    // Note: The tensor's home_device reflects the requested device
    // Actual GPU memory allocation happens via coherence (ensureOnDevice)
    EXPECT_EQ(tensor->home_device(), DeviceId::cuda(0));

    // Verify stats
    EXPECT_EQ(manager.getStats().gpu_buffer_count, 1);
    EXPECT_EQ(manager.getStats().cpu_buffer_count, 0);
    EXPECT_GT(manager.getStats().gpu_bytes_allocated, 0);
}

// =============================================================================
// allocateOnDevice Tests
// =============================================================================

TEST(Test__DomainAwareBufferManager, AllocateOnCPUDevice)
{
    DomainAwareBufferConfig config;
    DomainAwareBufferManager manager(config);

    auto *tensor = manager.allocateOnDevice(DeviceId::cpu(), "test_buffer",
                                            {64, 128}, BufferTensorType::FP32);

    ASSERT_NE(tensor, nullptr);
    EXPECT_TRUE(tensor->home_device().is_cpu());
    EXPECT_TRUE(manager.hasBuffer("test_buffer"));
    EXPECT_EQ(manager.getBuffer("test_buffer"), tensor);
}

TEST(Test__DomainAwareBufferManager, AllocateOnGPUDevice)
{
    DomainAwareBufferConfig config;
    DomainAwareBufferManager manager(config);

    auto *tensor = manager.allocateOnDevice(DeviceId::cuda(0), "gpu_buffer",
                                            {64, 128}, BufferTensorType::FP32);

    ASSERT_NE(tensor, nullptr);
    EXPECT_EQ(tensor->home_device(), DeviceId::cuda(0));
    EXPECT_TRUE(manager.hasBuffer("gpu_buffer"));
}

TEST(Test__DomainAwareBufferManager, AllocateOnROCmDevice)
{
    DomainAwareBufferConfig config;
    DomainAwareBufferManager manager(config);

    auto *tensor = manager.allocateOnDevice(DeviceId::rocm(0), "rocm_buffer",
                                            {64, 128}, BufferTensorType::FP32);

    ASSERT_NE(tensor, nullptr);
    EXPECT_EQ(tensor->home_device(), DeviceId::rocm(0));
}

// =============================================================================
// allocateNUMALocal Tests
// =============================================================================

TEST(Test__DomainAwareBufferManager, AllocateNUMALocalUsesNUMAAllocator)
{
    DomainAwareBufferConfig config;
    config.numa_allocator = &NUMAAllocator::instance();

    DomainAwareBufferManager manager(config);

    auto *tensor = manager.allocateNUMALocal(0, "numa_buffer", {256, 256}, BufferTensorType::FP32);

    ASSERT_NE(tensor, nullptr);
    EXPECT_TRUE(tensor->home_device().is_cpu());

    // Verify stats track NUMA allocation
    EXPECT_EQ(manager.getStats().cpu_buffer_count, 1);
    EXPECT_GT(manager.getStats().cpu_bytes_allocated, 0);
}

TEST(Test__DomainAwareBufferManager, AllocateNUMALocalWithoutAllocator)
{
    // Without NUMA allocator, should fall back to standard allocation
    DomainAwareBufferConfig config;
    config.numa_allocator = nullptr;

    DomainAwareBufferManager manager(config);

    auto *tensor = manager.allocateNUMALocal(-1, "fallback_buffer", {128, 128}, BufferTensorType::FP32);

    ASSERT_NE(tensor, nullptr);
    EXPECT_TRUE(tensor->home_device().is_cpu());
}

// =============================================================================
// getDeviceForLayer Tests
// =============================================================================

TEST(Test__DomainAwareBufferManager, GetDeviceForLayerDelegatesToConfig)
{
    // Create placement config: layers 0-3 on CPU, rest on CUDA:0
    auto placement = LayerPlacementConfig::cpuFirstLayers(4, 28, DeviceId::cuda(0));

    DomainAwareBufferConfig config;
    config.placement_config = &placement;

    DomainAwareBufferManager manager(config);

    // CPU layers
    EXPECT_EQ(manager.getDeviceForLayer(0), DeviceId::cpu());
    EXPECT_EQ(manager.getDeviceForLayer(1), DeviceId::cpu());
    EXPECT_EQ(manager.getDeviceForLayer(2), DeviceId::cpu());
    EXPECT_EQ(manager.getDeviceForLayer(3), DeviceId::cpu());

    // GPU layers
    EXPECT_EQ(manager.getDeviceForLayer(4), DeviceId::cuda(0));
    EXPECT_EQ(manager.getDeviceForLayer(10), DeviceId::cuda(0));
    EXPECT_EQ(manager.getDeviceForLayer(27), DeviceId::cuda(0));
}

TEST(Test__DomainAwareBufferManager, GetDeviceForLayerUsesDefaultWhenNoConfig)
{
    DomainAwareBufferConfig config;
    config.default_device = DeviceId::rocm(1);
    config.placement_config = nullptr; // No placement config

    DomainAwareBufferManager manager(config);

    // Should return default device for all layers
    EXPECT_EQ(manager.getDeviceForLayer(0), DeviceId::rocm(1));
    EXPECT_EQ(manager.getDeviceForLayer(5), DeviceId::rocm(1));
    EXPECT_EQ(manager.getDeviceForLayer(27), DeviceId::rocm(1));
}

// =============================================================================
// releaseAll Tests
// =============================================================================

TEST(Test__DomainAwareBufferManager, ReleaseAllFreesBuffers)
{
    DomainAwareBufferConfig config;
    DomainAwareBufferManager manager(config);

    // Allocate several buffers
    manager.allocateOnDevice(DeviceId::cpu(), "buf1", {64, 64}, BufferTensorType::FP32);
    manager.allocateOnDevice(DeviceId::cpu(), "buf2", {128, 128}, BufferTensorType::FP32);
    manager.allocateOnDevice(DeviceId::cuda(0), "buf3", {256, 256}, BufferTensorType::FP32);

    EXPECT_EQ(manager.bufferCount(), 3);
    EXPECT_TRUE(manager.hasBuffer("buf1"));
    EXPECT_TRUE(manager.hasBuffer("buf2"));
    EXPECT_TRUE(manager.hasBuffer("buf3"));

    // Release all
    manager.releaseAll();

    EXPECT_EQ(manager.bufferCount(), 0);
    EXPECT_FALSE(manager.hasBuffer("buf1"));
    EXPECT_FALSE(manager.hasBuffer("buf2"));
    EXPECT_FALSE(manager.hasBuffer("buf3"));
    EXPECT_EQ(manager.getStats().total_bytes(), 0);
    EXPECT_EQ(manager.getStats().total_buffers(), 0);
}

// =============================================================================
// Statistics Tests
// =============================================================================

TEST(Test__DomainAwareBufferManager, StatsTrackGPUAllocations)
{
    DomainAwareBufferConfig config;
    DomainAwareBufferManager manager(config);

    // Allocate GPU buffers
    manager.allocateOnDevice(DeviceId::cuda(0), "gpu1", {64, 64}, BufferTensorType::FP32);
    manager.allocateOnDevice(DeviceId::cuda(0), "gpu2", {128, 128}, BufferTensorType::FP32);

    const auto &stats = manager.getStats();
    EXPECT_EQ(stats.gpu_buffer_count, 2);
    EXPECT_EQ(stats.cpu_buffer_count, 0);
    EXPECT_GT(stats.gpu_bytes_allocated, 0);
    EXPECT_EQ(stats.cpu_bytes_allocated, 0);

    // Verify per-device tracking
    EXPECT_TRUE(stats.bytes_per_device.count(DeviceId::cuda(0)) > 0);
    EXPECT_TRUE(stats.buffers_per_device.count(DeviceId::cuda(0)) > 0);
    EXPECT_EQ(stats.buffers_per_device.at(DeviceId::cuda(0)), 2);
}

TEST(Test__DomainAwareBufferManager, StatsTrackCPUAllocations)
{
    DomainAwareBufferConfig config;
    DomainAwareBufferManager manager(config);

    // Allocate CPU buffers
    manager.allocateOnDevice(DeviceId::cpu(), "cpu1", {64, 64}, BufferTensorType::FP32);
    manager.allocateOnDevice(DeviceId::cpu(), "cpu2", {128, 128}, BufferTensorType::FP32);

    const auto &stats = manager.getStats();
    EXPECT_EQ(stats.cpu_buffer_count, 2);
    EXPECT_EQ(stats.gpu_buffer_count, 0);
    EXPECT_GT(stats.cpu_bytes_allocated, 0);
    EXPECT_EQ(stats.gpu_bytes_allocated, 0);

    // Verify per-device tracking
    EXPECT_TRUE(stats.bytes_per_device.count(DeviceId::cpu()) > 0);
    EXPECT_TRUE(stats.buffers_per_device.count(DeviceId::cpu()) > 0);
    EXPECT_EQ(stats.buffers_per_device.at(DeviceId::cpu()), 2);
}

TEST(Test__DomainAwareBufferManager, StatsTrackMixedAllocations)
{
    DomainAwareBufferConfig config;
    DomainAwareBufferManager manager(config);

    // Allocate mixed buffers
    manager.allocateOnDevice(DeviceId::cpu(), "cpu_buf", {64, 64}, BufferTensorType::FP32);
    manager.allocateOnDevice(DeviceId::cuda(0), "cuda_buf", {64, 64}, BufferTensorType::FP32);
    manager.allocateOnDevice(DeviceId::rocm(0), "rocm_buf", {64, 64}, BufferTensorType::FP32);

    const auto &stats = manager.getStats();
    EXPECT_EQ(stats.total_buffers(), 3);
    EXPECT_EQ(stats.cpu_buffer_count, 1);
    EXPECT_EQ(stats.gpu_buffer_count, 2); // CUDA + ROCm
}

// =============================================================================
// Multiple Layers Tests
// =============================================================================

TEST(Test__DomainAwareBufferManager, MultipleLayersAllocateCorrectly)
{
    // Create placement config: layers 0-1 on CPU, 2-3 on CUDA:0
    auto placement = LayerPlacementConfig::cpuFirstLayers(2, 4, DeviceId::cuda(0));

    DomainAwareBufferConfig config;
    config.placement_config = &placement;

    DomainAwareBufferManager manager(config);

    // Allocate for all 4 layers
    for (int i = 0; i < 4; ++i)
    {
        auto *tensor = manager.allocateForLayer(i, "hidden", {128, 896}, BufferTensorType::FP32);
        ASSERT_NE(tensor, nullptr) << "Failed to allocate for layer " << i;

        if (i < 2)
        {
            EXPECT_TRUE(tensor->home_device().is_cpu())
                << "Layer " << i << " should be on CPU";
        }
        else
        {
            EXPECT_TRUE(tensor->home_device().is_gpu())
                << "Layer " << i << " should be on GPU";
        }
    }

    EXPECT_EQ(manager.bufferCount(), 4);
    EXPECT_EQ(manager.getStats().cpu_buffer_count, 2);
    EXPECT_EQ(manager.getStats().gpu_buffer_count, 2);
}

// =============================================================================
// Data Type Tests
// =============================================================================

TEST(Test__DomainAwareBufferManager, AllocateFP16Buffer)
{
    DomainAwareBufferConfig config;
    DomainAwareBufferManager manager(config);

    auto *tensor = manager.allocateOnDevice(DeviceId::cpu(), "fp16_buf",
                                            {64, 128}, BufferTensorType::FP16);

    ASSERT_NE(tensor, nullptr);
    EXPECT_EQ(manager.bufferCount(), 1);
}

TEST(Test__DomainAwareBufferManager, AllocateBF16Buffer)
{
    DomainAwareBufferConfig config;
    DomainAwareBufferManager manager(config);

    auto *tensor = manager.allocateOnDevice(DeviceId::cpu(), "bf16_buf",
                                            {64, 128}, BufferTensorType::BF16);

    ASSERT_NE(tensor, nullptr);
    EXPECT_EQ(manager.bufferCount(), 1);
}

TEST(Test__DomainAwareBufferManager, AllocateQ8_1Buffer)
{
    DomainAwareBufferConfig config;
    DomainAwareBufferManager manager(config);

    auto *tensor = manager.allocateOnDevice(DeviceId::cpu(), "q8_buf",
                                            {64, 128}, BufferTensorType::Q8_1);

    ASSERT_NE(tensor, nullptr);
    EXPECT_EQ(manager.bufferCount(), 1);
}

TEST(Test__DomainAwareBufferManager, AllocateINT32Buffer)
{
    DomainAwareBufferConfig config;
    DomainAwareBufferManager manager(config);

    auto *tensor = manager.allocateOnDevice(DeviceId::cpu(), "int32_buf",
                                            {64, 128}, BufferTensorType::INT32);

    ASSERT_NE(tensor, nullptr);
    EXPECT_EQ(manager.bufferCount(), 1);
}

// =============================================================================
// Edge Cases
// =============================================================================

TEST(Test__DomainAwareBufferManager, DuplicateBufferReturnsExisting)
{
    DomainAwareBufferConfig config;
    DomainAwareBufferManager manager(config);

    auto *first = manager.allocateOnDevice(DeviceId::cpu(), "same_name",
                                           {64, 64}, BufferTensorType::FP32);
    auto *second = manager.allocateOnDevice(DeviceId::cpu(), "same_name",
                                            {128, 128}, BufferTensorType::FP32); // Different shape!

    // Should return the existing buffer, not create a new one
    EXPECT_EQ(first, second);
    EXPECT_EQ(manager.bufferCount(), 1);
}

TEST(Test__DomainAwareBufferManager, ResetStatsWorks)
{
    DomainAwareBufferConfig config;
    DomainAwareBufferManager manager(config);

    // Allocate some buffers
    manager.allocateOnDevice(DeviceId::cpu(), "buf1", {64, 64}, BufferTensorType::FP32);
    manager.allocateOnDevice(DeviceId::cuda(0), "buf2", {64, 64}, BufferTensorType::FP32);

    EXPECT_GT(manager.getStats().total_bytes(), 0);

    // Reset stats (note: this resets stats but keeps buffers)
    manager.resetStats();

    // Stats should be reset
    EXPECT_EQ(manager.getStats().total_bytes(), 0);
    EXPECT_EQ(manager.getStats().total_buffers(), 0);

    // Buffers should still exist
    EXPECT_EQ(manager.bufferCount(), 2);
    EXPECT_TRUE(manager.hasBuffer("buf1"));
    EXPECT_TRUE(manager.hasBuffer("buf2"));
}
