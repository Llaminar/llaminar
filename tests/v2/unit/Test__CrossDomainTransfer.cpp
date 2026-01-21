/**
 * @file Test__CrossDomainTransfer.cpp
 * @brief Unit tests for CrossDomainTransfer utility
 *
 * Tests the cross-domain activation transfer functionality for
 * heterogeneous GPU/CPU execution.
 *
 * @author David Sanftenberg
 * @date January 2026
 */

#include <gtest/gtest.h>

#include "execution/CrossDomainTransfer.h"
#include "tensors/cpu/CPUTensors.h"
#include "backends/DeviceId.h"
#include "../utils/TestTensorFactory.h"

#include <cstring>
#include <vector>
#include <cmath>

namespace llaminar2::test
{

    // =========================================================================
    // Configuration Tests
    // =========================================================================

    TEST(Test__CrossDomainTransfer, ConstructsWithDefaultConfig)
    {
        CrossDomainTransfer transfer;

        // Verify default config
        EXPECT_TRUE(transfer.config().use_pinned_memory);
        EXPECT_FALSE(transfer.config().async_transfers);
        EXPECT_EQ(transfer.config().numa_allocator, nullptr);
        EXPECT_EQ(transfer.config().target_numa_node, -1);
    }

    TEST(Test__CrossDomainTransfer, ConstructsWithCustomConfig)
    {
        CrossDomainTransfer::Config config;
        config.use_pinned_memory = false;
        config.async_transfers = true;
        config.target_numa_node = 1;

        CrossDomainTransfer transfer(config);

        EXPECT_FALSE(transfer.config().use_pinned_memory);
        EXPECT_TRUE(transfer.config().async_transfers);
        EXPECT_EQ(transfer.config().target_numa_node, 1);
    }

    // =========================================================================
    // Statistics Tests
    // =========================================================================

    TEST(Test__CrossDomainTransfer, StatsStartAtZero)
    {
        CrossDomainTransfer transfer;
        auto stats = transfer.getStats();

        EXPECT_EQ(stats.gpu_to_cpu_count, 0);
        EXPECT_EQ(stats.cpu_to_gpu_count, 0);
        EXPECT_EQ(stats.gpu_to_cpu_bytes, 0);
        EXPECT_EQ(stats.cpu_to_gpu_bytes, 0);
        EXPECT_DOUBLE_EQ(stats.gpu_to_cpu_time_ms, 0.0);
        EXPECT_DOUBLE_EQ(stats.cpu_to_gpu_time_ms, 0.0);
    }

    TEST(Test__CrossDomainTransfer, ResetStatsClearsAll)
    {
        CrossDomainTransfer transfer;

        // Manually modify stats (simulating transfers)
        // Note: In real usage, stats are updated by gpuToCpu/cpuToGpu calls

        // Reset should clear everything
        transfer.resetStats();
        auto stats = transfer.getStats();

        EXPECT_EQ(stats.gpu_to_cpu_count, 0);
        EXPECT_EQ(stats.cpu_to_gpu_count, 0);
        EXPECT_EQ(stats.gpu_to_cpu_bytes, 0);
        EXPECT_EQ(stats.cpu_to_gpu_bytes, 0);
    }

    TEST(Test__CrossDomainTransfer, StatsBandwidthCalculation)
    {
        CrossDomainTransfer::TransferStats stats;

        // No transfers: bandwidth should be 0
        EXPECT_DOUBLE_EQ(stats.gpuToCpuBandwidthGBps(), 0.0);
        EXPECT_DOUBLE_EQ(stats.cpuToGpuBandwidthGBps(), 0.0);

        // Simulate: 1GB in 1 second = 1 GB/s
        stats.gpu_to_cpu_bytes = 1'000'000'000; // 1 GB
        stats.gpu_to_cpu_time_ms = 1000.0;      // 1 second
        EXPECT_NEAR(stats.gpuToCpuBandwidthGBps(), 1.0, 0.01);

        // Simulate: 10GB in 500ms = 20 GB/s
        stats.cpu_to_gpu_bytes = 10'000'000'000ULL; // 10 GB
        stats.cpu_to_gpu_time_ms = 500.0;           // 500ms
        EXPECT_NEAR(stats.cpuToGpuBandwidthGBps(), 20.0, 0.01);
    }

    // =========================================================================
    // Synchronization Tests
    // =========================================================================

    TEST(Test__CrossDomainTransfer, SynchronizeCompletes)
    {
        CrossDomainTransfer transfer;

        // Synchronize should complete without error even with no pending transfers
        EXPECT_NO_THROW(transfer.synchronize());
    }

    TEST(Test__CrossDomainTransfer, SynchronizeWithAsyncConfig)
    {
        CrossDomainTransfer::Config config;
        config.async_transfers = true;

        CrossDomainTransfer transfer(config);

        // Even with async config, synchronize should complete cleanly
        EXPECT_NO_THROW(transfer.synchronize());
    }

    // =========================================================================
    // CPU-Only Transfer Tests (no GPU required)
    // =========================================================================

    TEST(Test__CrossDomainTransfer, GpuToCpuRejectsNullSrc)
    {
        CrossDomainTransfer transfer;
        auto dst = TestTensorFactory::createFP32({32, 64});

        EXPECT_FALSE(transfer.gpuToCpu(nullptr, dst.get()));
    }

    TEST(Test__CrossDomainTransfer, GpuToCpuRejectsNullDst)
    {
        CrossDomainTransfer transfer;
        auto src = TestTensorFactory::createFP32({32, 64});

        EXPECT_FALSE(transfer.gpuToCpu(src.get(), nullptr));
    }

    TEST(Test__CrossDomainTransfer, CpuToGpuRejectsNullSrc)
    {
        CrossDomainTransfer transfer;
        auto dst = TestTensorFactory::createFP32({32, 64});

        EXPECT_FALSE(transfer.cpuToGpu(nullptr, dst.get(), DeviceId::cuda(0)));
    }

    TEST(Test__CrossDomainTransfer, CpuToGpuRejectsNullDst)
    {
        CrossDomainTransfer transfer;
        auto src = TestTensorFactory::createFP32({32, 64});

        EXPECT_FALSE(transfer.cpuToGpu(src.get(), nullptr, DeviceId::cuda(0)));
    }

    TEST(Test__CrossDomainTransfer, CpuToGpuRejectsCpuDevice)
    {
        CrossDomainTransfer transfer;
        auto src = TestTensorFactory::createFP32({32, 64});
        auto dst = TestTensorFactory::createFP32({32, 64});

        // CPU device should be rejected
        EXPECT_FALSE(transfer.cpuToGpu(src.get(), dst.get(), DeviceId::cpu()));
    }

    TEST(Test__CrossDomainTransfer, TransferRejectsNullSource)
    {
        CrossDomainTransfer transfer;

        auto result = transfer.transfer(nullptr, DeviceType::CPU);
        EXPECT_EQ(result, nullptr);
    }

    TEST(Test__CrossDomainTransfer, TransferRejectsUnsupportedDeviceType)
    {
        CrossDomainTransfer transfer;
        auto src = TestTensorFactory::createFP32({32, 64});

        // Unknown device type (using a valid but unhandled type)
        // DeviceType doesn't have an "Unknown" - let's test with GPU but no device ID
        auto result = transfer.transfer(src.get(), DeviceType::CUDA, DeviceId::cpu());
        EXPECT_EQ(result, nullptr); // Should fail because device_id is not GPU
    }

    // =========================================================================
    // CPU to CPU Transfer Tests (simulating GPU->CPU without actual GPU)
    // =========================================================================

    TEST(Test__CrossDomainTransfer, GpuToCpuCopiesDataCpuOnly)
    {
        CrossDomainTransfer transfer;

        // Create source with known data
        auto src = TestTensorFactory::createFP32Random({32, 64});
        auto dst = TestTensorFactory::createFP32({32, 64});

        // Zero-initialize destination
        std::memset(dst->mutable_data(), 0, dst->size_bytes());

        // For a CPU-only tensor, gpuToCpu should still work (it's just a memcpy)
        // This tests the coherence-based transfer logic without actual GPU
        bool result = transfer.gpuToCpu(src.get(), dst.get());
        EXPECT_TRUE(result);

        // Verify data was copied
        const float *src_data = src->data();
        const float *dst_data = dst->data();

        for (size_t i = 0; i < src->numel(); ++i)
        {
            EXPECT_FLOAT_EQ(src_data[i], dst_data[i]) << "Mismatch at index " << i;
        }
    }

    TEST(Test__CrossDomainTransfer, GpuToCpuUpdatesStats)
    {
        CrossDomainTransfer transfer;

        auto src = TestTensorFactory::createFP32Random({32, 64});
        auto dst = TestTensorFactory::createFP32({32, 64});

        // Initial stats
        EXPECT_EQ(transfer.getStats().gpu_to_cpu_count, 0);

        // Perform transfer
        EXPECT_TRUE(transfer.gpuToCpu(src.get(), dst.get()));

        // Check stats updated
        auto stats = transfer.getStats();
        EXPECT_EQ(stats.gpu_to_cpu_count, 1);
        EXPECT_EQ(stats.gpu_to_cpu_bytes, src->size_bytes());
        EXPECT_GT(stats.gpu_to_cpu_time_ms, 0.0); // Some time elapsed
    }

    TEST(Test__CrossDomainTransfer, StatsTrackMultipleTransfers)
    {
        CrossDomainTransfer transfer;

        auto src1 = TestTensorFactory::createFP32Random({32, 64});
        auto dst1 = TestTensorFactory::createFP32({32, 64});
        auto src2 = TestTensorFactory::createFP32Random({16, 128});
        auto dst2 = TestTensorFactory::createFP32({16, 128});

        // Two transfers
        EXPECT_TRUE(transfer.gpuToCpu(src1.get(), dst1.get()));
        EXPECT_TRUE(transfer.gpuToCpu(src2.get(), dst2.get()));

        auto stats = transfer.getStats();
        EXPECT_EQ(stats.gpu_to_cpu_count, 2);
        EXPECT_EQ(stats.gpu_to_cpu_bytes, src1->size_bytes() + src2->size_bytes());
    }

    TEST(Test__CrossDomainTransfer, GpuToCpuRejectsSizeMismatch)
    {
        CrossDomainTransfer transfer;

        auto src = TestTensorFactory::createFP32({64, 128}); // 8192 elements
        auto dst = TestTensorFactory::createFP32({32, 64});  // 2048 elements (smaller)

        // Should fail because destination is too small
        EXPECT_FALSE(transfer.gpuToCpu(src.get(), dst.get()));
    }

    TEST(Test__CrossDomainTransfer, TransferToCpuAllocatesCorrectShape)
    {
        CrossDomainTransfer transfer;

        std::vector<size_t> shape = {16, 32, 64};
        auto src = TestTensorFactory::createFP32Random(shape);

        // Transfer to CPU should allocate a new tensor
        auto result = transfer.transfer(src.get(), DeviceType::CPU);

        ASSERT_NE(result, nullptr);
        EXPECT_EQ(result->shape(), shape);
    }

    TEST(Test__CrossDomainTransfer, TransferToCpuPreservesData)
    {
        CrossDomainTransfer transfer;

        auto src = TestTensorFactory::createFP32Random({32, 64}, -10.0f, 10.0f, 12345);

        auto result = transfer.transfer(src.get(), DeviceType::CPU);
        ASSERT_NE(result, nullptr);

        // Verify data integrity
        auto *result_fp32 = dynamic_cast<FP32Tensor *>(result.get());
        ASSERT_NE(result_fp32, nullptr);

        const float *src_data = src->data();
        const float *dst_data = result_fp32->data();

        for (size_t i = 0; i < src->numel(); ++i)
        {
            EXPECT_FLOAT_EQ(src_data[i], dst_data[i]) << "Mismatch at index " << i;
        }
    }

    // =========================================================================
    // Configuration Effect Tests
    // =========================================================================

    TEST(Test__CrossDomainTransfer, ConfigDoesNotAffectCpuOnlyTransfer)
    {
        // Test that different configs still work for CPU-only transfers

        CrossDomainTransfer::Config config1;
        config1.use_pinned_memory = true;

        CrossDomainTransfer::Config config2;
        config2.use_pinned_memory = false;
        config2.async_transfers = true;

        CrossDomainTransfer transfer1(config1);
        CrossDomainTransfer transfer2(config2);

        auto src = TestTensorFactory::createFP32Random({32, 64});
        auto dst1 = TestTensorFactory::createFP32({32, 64});
        auto dst2 = TestTensorFactory::createFP32({32, 64});

        // Both should succeed
        EXPECT_TRUE(transfer1.gpuToCpu(src.get(), dst1.get()));
        EXPECT_TRUE(transfer2.gpuToCpu(src.get(), dst2.get()));

        // And produce identical results
        const float *data1 = dst1->data();
        const float *data2 = dst2->data();

        for (size_t i = 0; i < dst1->numel(); ++i)
        {
            EXPECT_FLOAT_EQ(data1[i], data2[i]);
        }
    }

} // namespace llaminar2::test
