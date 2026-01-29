/**
 * @file Test__HeterogeneousBackend_Phase3.cpp
 * @brief Integration tests for HeterogeneousBackend Phase 3 (Intra-Domain Broadcast)
 *
 * These tests require actual GPU hardware and test the Phase 3 algorithm:
 * - NCCL broadcast from cuda:0 to all CUDA GPUs (if >1 CUDA device)
 * - RCCL broadcast from rocm:0 to all ROCm GPUs (if >1 ROCm device)
 * - Combined Phase 1 + Phase 2 + Phase 3 execution
 *
 * NOTE: This test uses global backend accessors for device allocation to avoid
 * directly including both cuda_runtime.h and hip_runtime.h (which conflict).
 *
 * Test naming: V2_Integration_HeterogeneousBackend_Phase3_*
 *
 * @author David Sanftenberg
 * @date January 2026
 */

#include <gtest/gtest.h>
#include "v2/collective/backends/HeterogeneousBackend.h"
#include "v2/collective/DeviceGroup.h"
#include "v2/backends/DeviceId.h"
#include "v2/backends/BackendManager.h"
#include "v2/backends/IBackend.h"

namespace llaminar2::test
{

    // ═══════════════════════════════════════════════════════════════════════════
    // Test Fixture
    // ═══════════════════════════════════════════════════════════════════════════

    class Test__HeterogeneousBackend_Phase3 : public ::testing::Test
    {
    protected:
        void SetUp() override
        {
            // Get backends via global accessors (avoids header conflicts)
            cuda_backend_ = getCUDABackend();
            rocm_backend_ = getROCmBackend();

            // Count available devices
            cuda_count_ = cuda_backend_ ? cuda_backend_->deviceCount() : 0;
            rocm_count_ = rocm_backend_ ? rocm_backend_->deviceCount() : 0;
        }

        void TearDown() override
        {
            // Cleanup any allocated buffers
            for (auto &[ptr, info] : allocated_cuda_buffers_)
            {
                if (cuda_backend_ && ptr)
                {
                    cuda_backend_->free(ptr, info.device_ordinal);
                }
            }
            allocated_cuda_buffers_.clear();

            for (auto &[ptr, info] : allocated_rocm_buffers_)
            {
                if (rocm_backend_ && ptr)
                {
                    rocm_backend_->free(ptr, info.device_ordinal);
                }
            }
            allocated_rocm_buffers_.clear();
        }

        // ─────────────────────────────────────────────────────────────────────
        // Helper: Allocate and track buffers
        // ─────────────────────────────────────────────────────────────────────

        struct BufferInfo
        {
            int device_ordinal;
            size_t size;
        };

        void *allocateCUDA(size_t bytes, int ordinal = 0)
        {
            void *ptr = cuda_backend_->allocate(bytes, ordinal);
            if (ptr)
            {
                allocated_cuda_buffers_[ptr] = {ordinal, bytes};
            }
            return ptr;
        }

        void *allocateROCm(size_t bytes, int ordinal = 0)
        {
            void *ptr = rocm_backend_->allocate(bytes, ordinal);
            if (ptr)
            {
                allocated_rocm_buffers_[ptr] = {ordinal, bytes};
            }
            return ptr;
        }

        // ─────────────────────────────────────────────────────────────────────
        // Helper: Transfer data
        // ─────────────────────────────────────────────────────────────────────

        bool hostToDeviceCUDA(void *dst, const void *src, size_t bytes, int ordinal = 0)
        {
            return cuda_backend_->hostToDevice(dst, src, bytes, ordinal);
        }

        bool deviceToHostCUDA(void *dst, const void *src, size_t bytes, int ordinal = 0)
        {
            return cuda_backend_->deviceToHost(dst, src, bytes, ordinal);
        }

        bool hostToDeviceROCm(void *dst, const void *src, size_t bytes, int ordinal = 0)
        {
            return rocm_backend_->hostToDevice(dst, src, bytes, ordinal);
        }

        bool deviceToHostROCm(void *dst, const void *src, size_t bytes, int ordinal = 0)
        {
            return rocm_backend_->deviceToHost(dst, src, bytes, ordinal);
        }

        IBackend *cuda_backend_ = nullptr;
        IBackend *rocm_backend_ = nullptr;
        int cuda_count_ = 0;
        int rocm_count_ = 0;

        std::unordered_map<void *, BufferInfo> allocated_cuda_buffers_;
        std::unordered_map<void *, BufferInfo> allocated_rocm_buffers_;
    };

    // ═══════════════════════════════════════════════════════════════════════════
    // Phase 3 Integration Tests
    // ═══════════════════════════════════════════════════════════════════════════

#if defined(HAVE_CUDA) && defined(HAVE_ROCM)

    TEST_F(Test__HeterogeneousBackend_Phase3, PlanShowsRCCLBroadcastWith2ROCm)
    {
        // Verify Phase 3 plan shows RCCL broadcast will be called with 2 ROCm devices
        if (cuda_count_ < 1 || rocm_count_ < 2)
        {
            GTEST_SKIP() << "Test requires at least 1 CUDA and 2 ROCm devices";
        }

        HeterogeneousBackend backend;

        DeviceGroupBuilder builder;
        auto group = builder
                         .setName("1_cuda_2_rocm")
                         .setScope(CollectiveScope::LOCAL)
                         .addDevice(DeviceId::cuda(0))
                         .addDevice(DeviceId::rocm(0))
                         .addDevice(DeviceId::rocm(1))
                         .setLocalRank(0)
                         .build();

        ASSERT_TRUE(backend.initialize(group));

        auto plan = backend.planPhase3();
        EXPECT_FALSE(plan.will_call_nccl_broadcast); // Only 1 CUDA device
        EXPECT_TRUE(plan.will_call_rccl_broadcast);  // 2 ROCm devices
        EXPECT_EQ(plan.nccl_broadcast_root, -1);     // No NCCL broadcast
        EXPECT_EQ(plan.rccl_broadcast_root, 0);      // rocm:0 is root
        EXPECT_EQ(plan.nccl_device_count, 1u);
        EXPECT_EQ(plan.rccl_device_count, 2u);

        backend.shutdown();
    }

    TEST_F(Test__HeterogeneousBackend_Phase3, PlanShowsNCCLBroadcastWith2CUDA)
    {
        // Verify Phase 3 plan shows NCCL broadcast will be called with 2 CUDA devices
        if (cuda_count_ < 2 || rocm_count_ < 1)
        {
            GTEST_SKIP() << "Test requires at least 2 CUDA and 1 ROCm device";
        }

        HeterogeneousBackend backend;

        DeviceGroupBuilder builder;
        auto group = builder
                         .setName("2_cuda_1_rocm")
                         .setScope(CollectiveScope::LOCAL)
                         .addDevice(DeviceId::cuda(0))
                         .addDevice(DeviceId::cuda(1))
                         .addDevice(DeviceId::rocm(0))
                         .setLocalRank(0)
                         .build();

        ASSERT_TRUE(backend.initialize(group));

        auto plan = backend.planPhase3();
        EXPECT_TRUE(plan.will_call_nccl_broadcast);  // 2 CUDA devices
        EXPECT_FALSE(plan.will_call_rccl_broadcast); // Only 1 ROCm device
        EXPECT_EQ(plan.nccl_broadcast_root, 0);      // cuda:0 is root
        EXPECT_EQ(plan.rccl_broadcast_root, -1);     // No RCCL broadcast
        EXPECT_EQ(plan.nccl_device_count, 2u);
        EXPECT_EQ(plan.rccl_device_count, 1u);

        backend.shutdown();
    }

    TEST_F(Test__HeterogeneousBackend_Phase3, PlanShowsNoBroadcastWithSingleDevices)
    {
        // Verify Phase 3 plan shows no broadcasts with 1 CUDA + 1 ROCm
        if (cuda_count_ < 1 || rocm_count_ < 1)
        {
            GTEST_SKIP() << "Test requires at least 1 CUDA and 1 ROCm device";
        }

        HeterogeneousBackend backend;

        DeviceGroupBuilder builder;
        auto group = builder
                         .setName("1_cuda_1_rocm")
                         .setScope(CollectiveScope::LOCAL)
                         .addDevice(DeviceId::cuda(0))
                         .addDevice(DeviceId::rocm(0))
                         .setLocalRank(0)
                         .build();

        ASSERT_TRUE(backend.initialize(group));

        auto plan = backend.planPhase3();
        EXPECT_FALSE(plan.will_call_nccl_broadcast); // Only 1 CUDA device
        EXPECT_FALSE(plan.will_call_rccl_broadcast); // Only 1 ROCm device
        EXPECT_EQ(plan.nccl_broadcast_root, -1);
        EXPECT_EQ(plan.rccl_broadcast_root, -1);
        EXPECT_EQ(plan.nccl_device_count, 1u);
        EXPECT_EQ(plan.rccl_device_count, 1u);

        backend.shutdown();
    }

    TEST_F(Test__HeterogeneousBackend_Phase3, RCCLBroadcastTo2AMD)
    {
        // Test Phase 3 only: RCCL broadcast from rocm:0 to rocm:1
        // rocm:0 = [10,20,30], rocm:1 = [0,0,0] (or garbage)
        // Expected result: both = [10,20,30]

        if (cuda_count_ < 1 || rocm_count_ < 2)
        {
            GTEST_SKIP() << "Test requires at least 1 CUDA and 2 ROCm devices";
        }

        const size_t count = 3;
        const size_t bytes = count * sizeof(float);

        // Allocate buffers
        void *cuda_buf = allocateCUDA(bytes, 0);
        void *rocm_buf_0 = allocateROCm(bytes, 0);
        void *rocm_buf_1 = allocateROCm(bytes, 1);

        ASSERT_NE(cuda_buf, nullptr) << "Failed to allocate CUDA buffer";
        ASSERT_NE(rocm_buf_0, nullptr) << "Failed to allocate ROCm buffer on device 0";
        ASSERT_NE(rocm_buf_1, nullptr) << "Failed to allocate ROCm buffer on device 1";

        // Initialize with known values
        float cuda_data[] = {0.0f, 0.0f, 0.0f};      // Dummy data for CUDA
        float rocm_data_0[] = {10.0f, 20.0f, 30.0f}; // Source for broadcast
        float rocm_data_1[] = {0.0f, 0.0f, 0.0f};    // Will receive broadcast

        ASSERT_TRUE(hostToDeviceCUDA(cuda_buf, cuda_data, bytes, 0));
        ASSERT_TRUE(hostToDeviceROCm(rocm_buf_0, rocm_data_0, bytes, 0));
        ASSERT_TRUE(hostToDeviceROCm(rocm_buf_1, rocm_data_1, bytes, 1));

        // Synchronize transfers
        cuda_backend_->synchronize(0);
        rocm_backend_->synchronize(0);
        rocm_backend_->synchronize(1);

        // Create and initialize HeterogeneousBackend
        HeterogeneousBackend backend;

        DeviceGroupBuilder builder;
        auto group = builder
                         .setName("1_cuda_2_rocm")
                         .setScope(CollectiveScope::LOCAL)
                         .addDevice(DeviceId::cuda(0))
                         .addDevice(DeviceId::rocm(0))
                         .addDevice(DeviceId::rocm(1))
                         .setLocalRank(0)
                         .build();

        ASSERT_TRUE(backend.initialize(group));

        // Verify RCCL backend was created (required for multi-ROCm)
        EXPECT_TRUE(backend.hasRCCLBackend());

        // Execute Phase 3 (intra-domain broadcast)
        std::vector<void *> cuda_buffers = {cuda_buf};
        std::vector<void *> rocm_buffers = {rocm_buf_0, rocm_buf_1};

        ASSERT_TRUE(backend.executePhase3_IntraDomainBroadcast(
            cuda_buffers, rocm_buffers, count,
            CollectiveDataType::FLOAT32));

        // Synchronize
        ASSERT_TRUE(backend.synchronize());

        // Read back results
        float rocm_result_0[3] = {0};
        float rocm_result_1[3] = {0};

        ASSERT_TRUE(deviceToHostROCm(rocm_result_0, rocm_buf_0, bytes, 0));
        ASSERT_TRUE(deviceToHostROCm(rocm_result_1, rocm_buf_1, bytes, 1));

        // Expected: both should have [10,20,30]
        float expected[] = {10.0f, 20.0f, 30.0f};

        for (size_t i = 0; i < count; ++i)
        {
            EXPECT_FLOAT_EQ(rocm_result_0[i], expected[i])
                << "ROCm:0 result mismatch at index " << i;
        }

        for (size_t i = 0; i < count; ++i)
        {
            EXPECT_FLOAT_EQ(rocm_result_1[i], expected[i])
                << "ROCm:1 result mismatch at index " << i;
        }

        backend.shutdown();
    }

    TEST_F(Test__HeterogeneousBackend_Phase3, BroadcastAfterPhase1And2)
    {
        // Test full 3-phase allreduce: Phase 1 + Phase 2 + Phase 3
        // cuda:0 = [1,1,1], rocm:0 = [2,2,2], rocm:1 = [3,3,3]
        // Phase 1: rocm:0 becomes [5,5,5] (sum of ROCm devices)
        // Phase 2: Both bridges become [6,6,6] (sum of cuda:0 + rocm:0)
        // Phase 3: rocm:1 receives [6,6,6] from rocm:0
        // Expected: ALL devices have [6,6,6]

        if (cuda_count_ < 1 || rocm_count_ < 2)
        {
            GTEST_SKIP() << "Test requires at least 1 CUDA and 2 ROCm devices";
        }

        const size_t count = 3;
        const size_t bytes = count * sizeof(float);

        // Allocate buffers
        void *cuda_buf_0 = allocateCUDA(bytes, 0);
        void *rocm_buf_0 = allocateROCm(bytes, 0);
        void *rocm_buf_1 = allocateROCm(bytes, 1);

        ASSERT_NE(cuda_buf_0, nullptr) << "Failed to allocate CUDA buffer on device 0";
        ASSERT_NE(rocm_buf_0, nullptr) << "Failed to allocate ROCm buffer on device 0";
        ASSERT_NE(rocm_buf_1, nullptr) << "Failed to allocate ROCm buffer on device 1";

        // Initialize with known values
        float cuda_data_0[] = {1.0f, 1.0f, 1.0f};
        float rocm_data_0[] = {2.0f, 2.0f, 2.0f};
        float rocm_data_1[] = {3.0f, 3.0f, 3.0f};

        ASSERT_TRUE(hostToDeviceCUDA(cuda_buf_0, cuda_data_0, bytes, 0));
        ASSERT_TRUE(hostToDeviceROCm(rocm_buf_0, rocm_data_0, bytes, 0));
        ASSERT_TRUE(hostToDeviceROCm(rocm_buf_1, rocm_data_1, bytes, 1));

        // Synchronize transfers
        cuda_backend_->synchronize(0);
        rocm_backend_->synchronize(0);
        rocm_backend_->synchronize(1);

        // Create and initialize HeterogeneousBackend
        HeterogeneousBackend backend;

        DeviceGroupBuilder builder;
        auto group = builder
                         .setName("1_cuda_2_rocm")
                         .setScope(CollectiveScope::LOCAL)
                         .addDevice(DeviceId::cuda(0))
                         .addDevice(DeviceId::rocm(0))
                         .addDevice(DeviceId::rocm(1))
                         .setLocalRank(0)
                         .build();

        ASSERT_TRUE(backend.initialize(group));

        // Prepare buffer vectors
        std::vector<void *> cuda_buffers = {cuda_buf_0};
        std::vector<void *> rocm_buffers = {rocm_buf_0, rocm_buf_1};

        // ─────────────────────────────────────────────────────────────────────
        // Phase 1: Intra-domain reduce
        // ─────────────────────────────────────────────────────────────────────
        ASSERT_TRUE(backend.executePhase1_IntraDomainReduce(
            cuda_buffers, rocm_buffers, count,
            CollectiveDataType::FLOAT32,
            CollectiveOp::ALLREDUCE_SUM));

        // After Phase 1: cuda:0 = [1,1,1], rocm:0 = [5,5,5], rocm:1 unchanged

        // ─────────────────────────────────────────────────────────────────────
        // Phase 2: Bridge exchange
        // ─────────────────────────────────────────────────────────────────────
        ASSERT_TRUE(backend.executePhase2_BridgeExchange(
            cuda_buf_0, rocm_buf_0, count,
            CollectiveDataType::FLOAT32,
            CollectiveOp::ALLREDUCE_SUM));

        // After Phase 2: cuda:0 = [6,6,6], rocm:0 = [6,6,6], rocm:1 still has old value

        // ─────────────────────────────────────────────────────────────────────
        // Phase 3: Intra-domain broadcast
        // ─────────────────────────────────────────────────────────────────────
        ASSERT_TRUE(backend.executePhase3_IntraDomainBroadcast(
            cuda_buffers, rocm_buffers, count,
            CollectiveDataType::FLOAT32));

        // Synchronize
        ASSERT_TRUE(backend.synchronize());

        // Read back results
        float cuda_result[3] = {0};
        float rocm_result_0[3] = {0};
        float rocm_result_1[3] = {0};

        ASSERT_TRUE(deviceToHostCUDA(cuda_result, cuda_buf_0, bytes, 0));
        ASSERT_TRUE(deviceToHostROCm(rocm_result_0, rocm_buf_0, bytes, 0));
        ASSERT_TRUE(deviceToHostROCm(rocm_result_1, rocm_buf_1, bytes, 1));

        // Expected: ALL devices have [6,6,6]
        float expected[] = {6.0f, 6.0f, 6.0f};

        for (size_t i = 0; i < count; ++i)
        {
            EXPECT_FLOAT_EQ(cuda_result[i], expected[i])
                << "CUDA:0 result mismatch at index " << i;
        }

        for (size_t i = 0; i < count; ++i)
        {
            EXPECT_FLOAT_EQ(rocm_result_0[i], expected[i])
                << "ROCm:0 result mismatch at index " << i;
        }

        for (size_t i = 0; i < count; ++i)
        {
            EXPECT_FLOAT_EQ(rocm_result_1[i], expected[i])
                << "ROCm:1 result mismatch at index " << i;
        }

        backend.shutdown();
    }

    TEST_F(Test__HeterogeneousBackend_Phase3, BroadcastWithLargeTensor)
    {
        // Test Phase 3 with a larger tensor (1MB)
        // This tests that the broadcast works correctly with large data

        if (cuda_count_ < 1 || rocm_count_ < 2)
        {
            GTEST_SKIP() << "Test requires at least 1 CUDA and 2 ROCm devices";
        }

        // 1MB = 262144 floats (1048576 bytes)
        const size_t count = 262144;
        const size_t bytes = count * sizeof(float);

        // Allocate buffers
        void *cuda_buf = allocateCUDA(bytes, 0);
        void *rocm_buf_0 = allocateROCm(bytes, 0);
        void *rocm_buf_1 = allocateROCm(bytes, 1);

        ASSERT_NE(cuda_buf, nullptr) << "Failed to allocate CUDA buffer";
        ASSERT_NE(rocm_buf_0, nullptr) << "Failed to allocate ROCm buffer on device 0";
        ASSERT_NE(rocm_buf_1, nullptr) << "Failed to allocate ROCm buffer on device 1";

        // Initialize with a pattern
        std::vector<float> cuda_data(count, 0.0f);
        std::vector<float> rocm_data_0(count);
        std::vector<float> rocm_data_1(count, 0.0f);

        // Fill rocm_data_0 with a recognizable pattern
        for (size_t i = 0; i < count; ++i)
        {
            rocm_data_0[i] = static_cast<float>(i % 1000); // Repeating pattern 0-999
        }

        ASSERT_TRUE(hostToDeviceCUDA(cuda_buf, cuda_data.data(), bytes, 0));
        ASSERT_TRUE(hostToDeviceROCm(rocm_buf_0, rocm_data_0.data(), bytes, 0));
        ASSERT_TRUE(hostToDeviceROCm(rocm_buf_1, rocm_data_1.data(), bytes, 1));

        // Synchronize transfers
        cuda_backend_->synchronize(0);
        rocm_backend_->synchronize(0);
        rocm_backend_->synchronize(1);

        // Create and initialize HeterogeneousBackend
        HeterogeneousBackend backend;

        DeviceGroupBuilder builder;
        auto group = builder
                         .setName("1_cuda_2_rocm")
                         .setScope(CollectiveScope::LOCAL)
                         .addDevice(DeviceId::cuda(0))
                         .addDevice(DeviceId::rocm(0))
                         .addDevice(DeviceId::rocm(1))
                         .setLocalRank(0)
                         .build();

        ASSERT_TRUE(backend.initialize(group));

        // Execute Phase 3 (intra-domain broadcast)
        std::vector<void *> cuda_buffers = {cuda_buf};
        std::vector<void *> rocm_buffers = {rocm_buf_0, rocm_buf_1};

        ASSERT_TRUE(backend.executePhase3_IntraDomainBroadcast(
            cuda_buffers, rocm_buffers, count,
            CollectiveDataType::FLOAT32));

        // Synchronize
        ASSERT_TRUE(backend.synchronize());

        // Read back result from rocm:1 (should now match rocm_data_0)
        std::vector<float> rocm_result_1(count);
        ASSERT_TRUE(deviceToHostROCm(rocm_result_1.data(), rocm_buf_1, bytes, 1));

        // Verify first 10 elements
        for (size_t i = 0; i < 10; ++i)
        {
            EXPECT_FLOAT_EQ(rocm_result_1[i], rocm_data_0[i])
                << "ROCm:1 result mismatch at index " << i;
        }

        // Verify last 10 elements
        for (size_t i = count - 10; i < count; ++i)
        {
            EXPECT_FLOAT_EQ(rocm_result_1[i], rocm_data_0[i])
                << "ROCm:1 result mismatch at index " << i;
        }

        // Verify middle element
        EXPECT_FLOAT_EQ(rocm_result_1[count / 2], rocm_data_0[count / 2]);

        backend.shutdown();
    }

    TEST_F(Test__HeterogeneousBackend_Phase3, Phase3FailsWhenNotInitialized)
    {
        // executePhase3 should fail if backend is not initialized
        HeterogeneousBackend backend;

        std::vector<void *> cuda_buffers = {reinterpret_cast<void *>(0x1000)};
        std::vector<void *> rocm_buffers = {reinterpret_cast<void *>(0x2000)};

        EXPECT_FALSE(backend.executePhase3_IntraDomainBroadcast(
            cuda_buffers, rocm_buffers, 4,
            CollectiveDataType::FLOAT32));
    }

    TEST_F(Test__HeterogeneousBackend_Phase3, Phase3FailsWithBufferCountMismatch)
    {
        // executePhase3 should fail if buffer counts don't match device counts
        if (cuda_count_ < 1 || rocm_count_ < 2)
        {
            GTEST_SKIP() << "Test requires at least 1 CUDA and 2 ROCm devices";
        }

        const size_t count = 4;
        const size_t bytes = count * sizeof(float);

        void *cuda_buf = allocateCUDA(bytes, 0);
        void *rocm_buf_0 = allocateROCm(bytes, 0);

        ASSERT_NE(cuda_buf, nullptr);
        ASSERT_NE(rocm_buf_0, nullptr);

        HeterogeneousBackend backend;

        DeviceGroupBuilder builder;
        auto group = builder
                         .setName("1_cuda_2_rocm")
                         .setScope(CollectiveScope::LOCAL)
                         .addDevice(DeviceId::cuda(0))
                         .addDevice(DeviceId::rocm(0))
                         .addDevice(DeviceId::rocm(1))
                         .setLocalRank(0)
                         .build();

        ASSERT_TRUE(backend.initialize(group));

        // Provide only 1 ROCm buffer when 2 are expected
        std::vector<void *> cuda_buffers = {cuda_buf};
        std::vector<void *> rocm_buffers = {rocm_buf_0}; // Missing second buffer!

        EXPECT_FALSE(backend.executePhase3_IntraDomainBroadcast(
            cuda_buffers, rocm_buffers, count,
            CollectiveDataType::FLOAT32));

        backend.shutdown();
    }

    TEST_F(Test__HeterogeneousBackend_Phase3, Phase3SkipsBroadcastWithMinimalConfig)
    {
        // Test that Phase 3 succeeds but does nothing with 1 CUDA + 1 ROCm
        // (No broadcasts needed - each domain already has the result)

        if (cuda_count_ < 1 || rocm_count_ < 1)
        {
            GTEST_SKIP() << "Test requires at least 1 CUDA and 1 ROCm device";
        }

        const size_t count = 4;
        const size_t bytes = count * sizeof(float);

        void *cuda_buf = allocateCUDA(bytes, 0);
        void *rocm_buf = allocateROCm(bytes, 0);

        ASSERT_NE(cuda_buf, nullptr);
        ASSERT_NE(rocm_buf, nullptr);

        // Initialize with known values
        float cuda_data[] = {1.0f, 2.0f, 3.0f, 4.0f};
        float rocm_data[] = {5.0f, 6.0f, 7.0f, 8.0f};

        ASSERT_TRUE(hostToDeviceCUDA(cuda_buf, cuda_data, bytes, 0));
        ASSERT_TRUE(hostToDeviceROCm(rocm_buf, rocm_data, bytes, 0));

        cuda_backend_->synchronize(0);
        rocm_backend_->synchronize(0);

        HeterogeneousBackend backend;

        DeviceGroupBuilder builder;
        auto group = builder
                         .setName("1_cuda_1_rocm")
                         .setScope(CollectiveScope::LOCAL)
                         .addDevice(DeviceId::cuda(0))
                         .addDevice(DeviceId::rocm(0))
                         .setLocalRank(0)
                         .build();

        ASSERT_TRUE(backend.initialize(group));

        // Verify no broadcasts will be called
        auto plan = backend.planPhase3();
        EXPECT_FALSE(plan.will_call_nccl_broadcast);
        EXPECT_FALSE(plan.will_call_rccl_broadcast);

        // Execute Phase 3 (should succeed but do nothing)
        std::vector<void *> cuda_buffers = {cuda_buf};
        std::vector<void *> rocm_buffers = {rocm_buf};

        ASSERT_TRUE(backend.executePhase3_IntraDomainBroadcast(
            cuda_buffers, rocm_buffers, count,
            CollectiveDataType::FLOAT32));

        ASSERT_TRUE(backend.synchronize());

        // Verify data is unchanged
        float cuda_result[4] = {0};
        float rocm_result[4] = {0};

        ASSERT_TRUE(deviceToHostCUDA(cuda_result, cuda_buf, bytes, 0));
        ASSERT_TRUE(deviceToHostROCm(rocm_result, rocm_buf, bytes, 0));

        for (size_t i = 0; i < count; ++i)
        {
            EXPECT_FLOAT_EQ(cuda_result[i], cuda_data[i])
                << "CUDA data changed at index " << i;
            EXPECT_FLOAT_EQ(rocm_result[i], rocm_data[i])
                << "ROCm data changed at index " << i;
        }

        backend.shutdown();
    }

#else

    TEST_F(Test__HeterogeneousBackend_Phase3, SkipWhenNoCUDAOrROCm)
    {
        GTEST_SKIP() << "HeterogeneousBackend Phase 3 tests require both HAVE_CUDA and HAVE_ROCM";
    }

#endif // defined(HAVE_CUDA) && defined(HAVE_ROCM)

} // namespace llaminar2::test
