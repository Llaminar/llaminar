/**
 * @file Test__CrossDomainTransferGPU.cpp
 * @brief GPU integration tests for CrossDomainTransfer utility
 *
 * These tests require actual GPU hardware (CUDA or ROCm) to run.
 * They verify data integrity across actual PCIe transfers.
 *
 * @author David Sanftenberg
 * @date January 2026
 */

#include <gtest/gtest.h>

#include "execution/CrossDomainTransfer.h"
#include "tensors/TensorClasses.h"
#include "backends/DeviceId.h"
#include "backends/ComputeBackend.h"
#include "../utils/TestTensorFactory.h"

#include <cstring>
#include <vector>
#include <cmath>

namespace llaminar2::test
{

    // =========================================================================
    // Test Fixture - Requires GPU
    // =========================================================================

    /**
     * @brief Test fixture that skips tests if no GPU is available
     */
    class CrossDomainTransferGPUTest : public ::testing::Test
    {
    protected:
        void SetUp() override
        {
            // Check for available GPU
            auto &dm = DeviceManager::instance();
            gpu_available_ = dm.has_gpu();

            if (!gpu_available_)
            {
                GTEST_SKIP() << "No GPU available, skipping GPU transfer tests";
            }

            // Get the first available GPU
#ifdef HAVE_CUDA
            if (dm.cuda_device_count() > 0)
            {
                gpu_device_ = DeviceId::cuda(dm.get_device_id_for_type(ComputeBackendType::GPU_CUDA, 0));
                return;
            }
#endif
#ifdef HAVE_ROCM
            if (dm.rocm_device_count() > 0)
            {
                gpu_device_ = DeviceId::rocm(dm.get_device_id_for_type(ComputeBackendType::GPU_ROCM, 0));
                return;
            }
#endif
            // Should not reach here if gpu_available_ is true
            gpu_device_ = DeviceId::cpu();
        }

        void TearDown() override
        {
            // Nothing to clean up
        }

        bool gpu_available_ = false;
        DeviceId gpu_device_ = DeviceId::cpu();
    };

    // =========================================================================
    // GPU Transfer Tests
    // =========================================================================

    TEST_F(CrossDomainTransferGPUTest, CpuToGpuBasicTransfer)
    {
        CrossDomainTransfer transfer;

        // Create CPU source with known data
        auto src = TestTensorFactory::createFP32Random({32, 64}, -1.0f, 1.0f, 42);

        // Create destination tensor (will be allocated on GPU)
        auto dst = TestTensorFactory::createFP32({32, 64});

        // Transfer to GPU
        bool result = transfer.cpuToGpu(src.get(), dst.get(), gpu_device_);
        EXPECT_TRUE(result);

        // Check stats
        auto stats = transfer.getStats();
        EXPECT_EQ(stats.cpu_to_gpu_count, 1);
        EXPECT_EQ(stats.cpu_to_gpu_bytes, src->size_bytes());
        EXPECT_GT(stats.cpu_to_gpu_time_ms, 0.0);
    }

    TEST_F(CrossDomainTransferGPUTest, GpuToCpuBasicTransfer)
    {
        CrossDomainTransfer transfer;

        // Create and upload source to GPU
        auto src = TestTensorFactory::createFP32Random({32, 64}, -1.0f, 1.0f, 42);
        src->ensureOnDevice(gpu_device_);
        src->mark_device_dirty(); // Mark GPU as having the data

        // Create destination
        auto dst = TestTensorFactory::createFP32({32, 64});
        std::memset(dst->mutable_data(), 0, dst->size_bytes());

        // Transfer back to CPU
        bool result = transfer.gpuToCpu(src.get(), dst.get());
        EXPECT_TRUE(result);

        // Verify data integrity
        const float *src_data = src->data(); // This syncs from GPU
        const float *dst_data = dst->data();

        for (size_t i = 0; i < src->numel(); ++i)
        {
            EXPECT_NEAR(src_data[i], dst_data[i], 1e-6f) << "Mismatch at index " << i;
        }
    }

    TEST_F(CrossDomainTransferGPUTest, RoundTripPreservesData)
    {
        CrossDomainTransfer transfer;

        // Create source with diverse values
        auto original = TestTensorFactory::createFP32Random({64, 128}, -100.0f, 100.0f, 12345);

        // Store original data for comparison
        std::vector<float> original_data(original->numel());
        std::memcpy(original_data.data(), original->data(), original->size_bytes());

        // Transfer CPU → GPU
        auto gpu_tensor = TestTensorFactory::createFP32({64, 128});
        ASSERT_TRUE(transfer.cpuToGpu(original.get(), gpu_tensor.get(), gpu_device_));

        // Transfer GPU → CPU (back)
        auto roundtrip = TestTensorFactory::createFP32({64, 128});
        ASSERT_TRUE(transfer.gpuToCpu(gpu_tensor.get(), roundtrip.get()));

        // Verify roundtrip preserved data
        const float *roundtrip_data = roundtrip->data();

        float max_diff = 0.0f;
        for (size_t i = 0; i < original->numel(); ++i)
        {
            float diff = std::abs(original_data[i] - roundtrip_data[i]);
            max_diff = std::max(max_diff, diff);
            EXPECT_NEAR(original_data[i], roundtrip_data[i], 1e-6f)
                << "Mismatch at index " << i
                << ": original=" << original_data[i]
                << ", roundtrip=" << roundtrip_data[i];
        }

        // Also check stats
        auto stats = transfer.getStats();
        EXPECT_EQ(stats.cpu_to_gpu_count, 1);
        EXPECT_EQ(stats.gpu_to_cpu_count, 1);

        GTEST_LOG_(INFO) << "Roundtrip max difference: " << max_diff;
    }

    TEST_F(CrossDomainTransferGPUTest, LargeTensorTransfer)
    {
        CrossDomainTransfer transfer;

        // Large tensor: 1024 x 4096 = 4M elements = 16MB
        const size_t rows = 1024;
        const size_t cols = 4096;

        auto src = TestTensorFactory::createFP32Random({rows, cols}, -1.0f, 1.0f, 999);
        auto gpu_dst = TestTensorFactory::createFP32({rows, cols});

        // Time the transfer
        auto start = std::chrono::high_resolution_clock::now();
        ASSERT_TRUE(transfer.cpuToGpu(src.get(), gpu_dst.get(), gpu_device_));
        auto end = std::chrono::high_resolution_clock::now();

        double elapsed_ms = std::chrono::duration<double, std::milli>(end - start).count();
        double size_mb = src->size_bytes() / (1024.0 * 1024.0);
        double bandwidth_gbps = (size_mb / 1024.0) / (elapsed_ms / 1000.0);

        GTEST_LOG_(INFO) << "Large transfer: " << size_mb << " MB in "
                         << elapsed_ms << " ms (" << bandwidth_gbps << " GB/s)";

        // Verify data on roundtrip
        auto cpu_dst = TestTensorFactory::createFP32({rows, cols});
        ASSERT_TRUE(transfer.gpuToCpu(gpu_dst.get(), cpu_dst.get()));

        // Spot-check values
        const float *src_data = src->data();
        const float *dst_data = cpu_dst->data();

        for (size_t i = 0; i < 1000; i += 100)
        {
            EXPECT_NEAR(src_data[i], dst_data[i], 1e-6f) << "Spot check failed at " << i;
        }
    }

    TEST_F(CrossDomainTransferGPUTest, MultipleTransfersSequential)
    {
        CrossDomainTransfer transfer;

        const int num_transfers = 5;
        std::vector<std::unique_ptr<FP32Tensor>> sources;
        std::vector<std::unique_ptr<FP32Tensor>> destinations;

        // Create multiple tensors
        for (int i = 0; i < num_transfers; ++i)
        {
            sources.push_back(TestTensorFactory::createFP32Random({32, 64}, -1.0f, 1.0f, 1000 + i));
            destinations.push_back(TestTensorFactory::createFP32({32, 64}));
        }

        // Sequential CPU → GPU transfers
        for (int i = 0; i < num_transfers; ++i)
        {
            ASSERT_TRUE(transfer.cpuToGpu(sources[i].get(), destinations[i].get(), gpu_device_))
                << "Transfer " << i << " failed";
        }

        // Verify stats
        auto stats = transfer.getStats();
        EXPECT_EQ(stats.cpu_to_gpu_count, num_transfers);

        // Verify all data
        for (int i = 0; i < num_transfers; ++i)
        {
            auto result = TestTensorFactory::createFP32({32, 64});
            ASSERT_TRUE(transfer.gpuToCpu(destinations[i].get(), result.get()));

            const float *src_data = sources[i]->data();
            const float *result_data = result->data();

            for (size_t j = 0; j < sources[i]->numel(); ++j)
            {
                EXPECT_NEAR(src_data[j], result_data[j], 1e-6f)
                    << "Tensor " << i << ", index " << j;
            }
        }
    }

    TEST_F(CrossDomainTransferGPUTest, TransferAllocatesGpuTensor)
    {
        CrossDomainTransfer transfer;

        auto src = TestTensorFactory::createFP32Random({32, 64});

        // Transfer to GPU with automatic allocation
        auto result = transfer.transfer(src.get(), gpu_device_.type, gpu_device_);

        ASSERT_NE(result, nullptr);
        EXPECT_EQ(result->shape(), src->shape());

        // Verify data by transferring back
        auto roundtrip = TestTensorFactory::createFP32({32, 64});
        ASSERT_TRUE(transfer.gpuToCpu(result.get(), roundtrip.get()));

        const float *src_data = src->data();
        const float *dst_data = roundtrip->data();

        for (size_t i = 0; i < src->numel(); ++i)
        {
            EXPECT_NEAR(src_data[i], dst_data[i], 1e-6f);
        }
    }

    TEST_F(CrossDomainTransferGPUTest, TransferAllocatesCpuTensorFromGpu)
    {
        CrossDomainTransfer transfer;

        // Create tensor and upload to GPU
        auto gpu_src = TestTensorFactory::createFP32Random({32, 64});
        gpu_src->ensureOnDevice(gpu_device_);
        gpu_src->mark_device_dirty();

        // Transfer from GPU to CPU with automatic allocation
        auto result = transfer.transfer(gpu_src.get(), DeviceType::CPU);

        ASSERT_NE(result, nullptr);
        EXPECT_EQ(result->shape(), gpu_src->shape());

        // Verify data integrity
        const float *src_data = gpu_src->data(); // Syncs from GPU
        auto *result_fp32 = dynamic_cast<FP32Tensor *>(result.get());
        ASSERT_NE(result_fp32, nullptr);
        const float *dst_data = result_fp32->data();

        for (size_t i = 0; i < gpu_src->numel(); ++i)
        {
            EXPECT_NEAR(src_data[i], dst_data[i], 1e-6f);
        }
    }

    TEST_F(CrossDomainTransferGPUTest, PinnedMemoryConfigAffectsPerformance)
    {
        // This test verifies that pinned memory configuration is honored
        // Performance difference may be subtle on small transfers

        CrossDomainTransfer::Config pinned_config;
        pinned_config.use_pinned_memory = true;

        CrossDomainTransfer::Config unpinned_config;
        unpinned_config.use_pinned_memory = false;

        CrossDomainTransfer pinned_transfer(pinned_config);
        CrossDomainTransfer unpinned_transfer(unpinned_config);

        // Use a reasonably large tensor to see performance effects
        auto src = TestTensorFactory::createFP32Random({256, 256});
        auto dst_pinned = TestTensorFactory::createFP32({256, 256});
        auto dst_unpinned = TestTensorFactory::createFP32({256, 256});

        // Both should succeed
        EXPECT_TRUE(pinned_transfer.cpuToGpu(src.get(), dst_pinned.get(), gpu_device_));
        EXPECT_TRUE(unpinned_transfer.cpuToGpu(src.get(), dst_unpinned.get(), gpu_device_));

        // Log performance for analysis
        auto pinned_stats = pinned_transfer.getStats();
        auto unpinned_stats = unpinned_transfer.getStats();

        GTEST_LOG_(INFO) << "Pinned transfer: " << pinned_stats.cpu_to_gpu_time_ms << " ms";
        GTEST_LOG_(INFO) << "Unpinned transfer: " << unpinned_stats.cpu_to_gpu_time_ms << " ms";

        // Data should be identical regardless of pinning
        auto result_pinned = TestTensorFactory::createFP32({256, 256});
        auto result_unpinned = TestTensorFactory::createFP32({256, 256});

        ASSERT_TRUE(pinned_transfer.gpuToCpu(dst_pinned.get(), result_pinned.get()));
        ASSERT_TRUE(unpinned_transfer.gpuToCpu(dst_unpinned.get(), result_unpinned.get()));

        const float *data_pinned = result_pinned->data();
        const float *data_unpinned = result_unpinned->data();

        for (size_t i = 0; i < 100; ++i)
        {
            EXPECT_NEAR(data_pinned[i], data_unpinned[i], 1e-6f);
        }
    }

} // namespace llaminar2::test
