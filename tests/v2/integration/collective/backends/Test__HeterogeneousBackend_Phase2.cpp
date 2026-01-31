/**
 * @file Test__HeterogeneousBackend_Phase2.cpp
 * @brief Integration tests for HeterogeneousBackend Phase 2 (Bridge Exchange)
 *
 * These tests require actual GPU hardware and test the Phase 2 algorithm:
 * - PCIeBAR allreduce between cuda:0 and rocm:0 bridge devices
 * - Combined Phase 1 + Phase 2 execution
 *
 * NOTE: This test uses global backend accessors for device allocation to avoid
 * directly including both cuda_runtime.h and hip_runtime.h (which conflict).
 *
 * Test naming: V2_Integration_HeterogeneousBackend_Phase2_*
 *
 * @author David Sanftenberg
 * @date January 2026
 */

#include <gtest/gtest.h>
#include "v2/collective/backends/HeterogeneousBackend.h"
#include "v2/collective/backends/PCIeBARBackend.h"
#include "v2/collective/DeviceGroup.h"
#include "v2/backends/DeviceId.h"
#include "v2/backends/BackendManager.h"
#include "v2/backends/IBackend.h"

namespace llaminar2::test
{

    // ═══════════════════════════════════════════════════════════════════════════
    // Test Fixture
    // ═══════════════════════════════════════════════════════════════════════════

    class Test__HeterogeneousBackend_Phase2 : public ::testing::Test
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
    // Phase 2 Integration Tests
    // ═══════════════════════════════════════════════════════════════════════════

#if defined(HAVE_CUDA) && defined(HAVE_ROCM)

    TEST_F(Test__HeterogeneousBackend_Phase2, PlanShowsBridgeAllreduce)
    {
        // Verify Phase 2 plan shows bridge allreduce will be called after init
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

        auto plan = backend.planPhase2();
        EXPECT_TRUE(plan.will_call_bridge_allreduce);

        backend.shutdown();
    }

    TEST_F(Test__HeterogeneousBackend_Phase2, PlanShowsCorrectBridgeDevices)
    {
        // Verify plan shows cuda:0 and rocm:0 as bridges
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

        auto plan = backend.planPhase2();
        EXPECT_EQ(plan.cuda_bridge_device, DeviceId::cuda(0));
        EXPECT_EQ(plan.rocm_bridge_device, DeviceId::rocm(0));

        backend.shutdown();
    }

    TEST_F(Test__HeterogeneousBackend_Phase2, BridgeBackendIsInitialized)
    {
        // Verify bridge_backend_ is ready after initialize()
        if (cuda_count_ < 1 || rocm_count_ < 1)
        {
            GTEST_SKIP() << "Test requires at least 1 CUDA and 1 ROCm device";
        }

        HeterogeneousBackend backend;

        // Before init, no bridge backend
        EXPECT_FALSE(backend.hasBridgeBackend());

        DeviceGroupBuilder builder;
        auto group = builder
                         .setName("1_cuda_1_rocm")
                         .setScope(CollectiveScope::LOCAL)
                         .addDevice(DeviceId::cuda(0))
                         .addDevice(DeviceId::rocm(0))
                         .setLocalRank(0)
                         .build();

        ASSERT_TRUE(backend.initialize(group));

        // After init, bridge backend should exist
        EXPECT_TRUE(backend.hasBridgeBackend());

        backend.shutdown();
    }

    TEST_F(Test__HeterogeneousBackend_Phase2, BridgeExchangeBetweenCUDAAndROCm)
    {
        // Test Phase 2 only: bridge exchange between cuda:0 and rocm:0
        // cuda:0 = [1,2,3,4], rocm:0 = [10,20,30,40]
        // Expected result: both = [11,22,33,44]

        if (cuda_count_ < 1 || rocm_count_ < 1)
        {
            GTEST_SKIP() << "Test requires at least 1 CUDA and 1 ROCm device";
        }

        const size_t count = 4;
        const size_t bytes = count * sizeof(float);

        // Allocate buffers
        void *cuda_buf = allocateCUDA(bytes, 0);
        void *rocm_buf = allocateROCm(bytes, 0);

        ASSERT_NE(cuda_buf, nullptr) << "Failed to allocate CUDA buffer";
        ASSERT_NE(rocm_buf, nullptr) << "Failed to allocate ROCm buffer";

        // Initialize with known values
        float cuda_data[] = {1.0f, 2.0f, 3.0f, 4.0f};
        float rocm_data[] = {10.0f, 20.0f, 30.0f, 40.0f};

        ASSERT_TRUE(hostToDeviceCUDA(cuda_buf, cuda_data, bytes, 0));
        ASSERT_TRUE(hostToDeviceROCm(rocm_buf, rocm_data, bytes, 0));

        // Synchronize transfers
        cuda_backend_->synchronize(0);
        rocm_backend_->synchronize(0);

        // Create and initialize HeterogeneousBackend
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
        ASSERT_TRUE(backend.hasBridgeBackend());

        // Execute Phase 2 (bridge exchange)
        ASSERT_TRUE(backend.executePhase2_BridgeExchange(
            cuda_buf, rocm_buf, count,
            CollectiveDataType::FLOAT32,
            CollectiveOp::ALLREDUCE_SUM));

        // Synchronize
        ASSERT_TRUE(backend.synchronize());

        // Read back results
        float cuda_result[4] = {0};
        float rocm_result[4] = {0};

        ASSERT_TRUE(deviceToHostCUDA(cuda_result, cuda_buf, bytes, 0));
        ASSERT_TRUE(deviceToHostROCm(rocm_result, rocm_buf, bytes, 0));

        // Verify CUDA buffer has sum
        float expected[] = {11.0f, 22.0f, 33.0f, 44.0f};
        for (size_t i = 0; i < count; ++i)
        {
            EXPECT_FLOAT_EQ(cuda_result[i], expected[i])
                << "CUDA result mismatch at index " << i;
        }

        // Verify ROCm buffer has sum
        for (size_t i = 0; i < count; ++i)
        {
            EXPECT_FLOAT_EQ(rocm_result[i], expected[i])
                << "ROCm result mismatch at index " << i;
        }

        backend.shutdown();
    }

    TEST_F(Test__HeterogeneousBackend_Phase2, BridgeExchangeAfterPhase1)
    {
        // Test Phase 1 + Phase 2 combined with 1 CUDA + 2 ROCm
        // cuda:0 = [1,1,1], rocm:0 = [2,2,2], rocm:1 = [3,3,3]
        // Phase 1: rocm:0 becomes [5,5,5] (sum of rocm:0 + rocm:1)
        // Phase 2: both bridges become [6,6,6] (sum of cuda:0 + rocm:0)

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

        // Verify Phase 1 plan
        auto phase1_plan = backend.planPhase1();
        EXPECT_FALSE(phase1_plan.will_call_nccl_reduce); // Only 1 CUDA
        EXPECT_TRUE(phase1_plan.will_call_rccl_reduce);  // 2 ROCm devices

        // Execute Phase 1 (intra-domain reduce)
        std::vector<void *> cuda_buffers = {cuda_buf_0};
        std::vector<void *> rocm_buffers = {rocm_buf_0, rocm_buf_1};

        ASSERT_TRUE(backend.executePhase1_IntraDomainReduce(
            cuda_buffers, rocm_buffers, count,
            CollectiveDataType::FLOAT32,
            CollectiveOp::ALLREDUCE_SUM));

        // After Phase 1, rocm:0 should have [5,5,5] (sum of ROCm devices)
        // cuda:0 should still have [1,1,1] (no NCCL reduce with single CUDA)

        // Execute Phase 2 (bridge exchange)
        ASSERT_TRUE(backend.executePhase2_BridgeExchange(
            cuda_buf_0, rocm_buf_0, count,
            CollectiveDataType::FLOAT32,
            CollectiveOp::ALLREDUCE_SUM));

        // Synchronize
        ASSERT_TRUE(backend.synchronize());

        // Read back results
        float cuda_result[3] = {0};
        float rocm_result_0[3] = {0};

        ASSERT_TRUE(deviceToHostCUDA(cuda_result, cuda_buf_0, bytes, 0));
        ASSERT_TRUE(deviceToHostROCm(rocm_result_0, rocm_buf_0, bytes, 0));

        // Expected: cuda:0 + rocm:0 (after Phase 1) = [1,1,1] + [5,5,5] = [6,6,6]
        float expected[] = {6.0f, 6.0f, 6.0f};

        for (size_t i = 0; i < count; ++i)
        {
            EXPECT_FLOAT_EQ(cuda_result[i], expected[i])
                << "CUDA result mismatch at index " << i;
        }

        for (size_t i = 0; i < count; ++i)
        {
            EXPECT_FLOAT_EQ(rocm_result_0[i], expected[i])
                << "ROCm:0 result mismatch at index " << i;
        }

        backend.shutdown();
    }

    TEST_F(Test__HeterogeneousBackend_Phase2, LargeTensorBridgeExchange)
    {
        // Test with 1MB+ tensor to verify pipelined path in PCIeBARBackend
        // This tests that the bridge exchange works correctly with large data

        if (cuda_count_ < 1 || rocm_count_ < 1)
        {
            GTEST_SKIP() << "Test requires at least 1 CUDA and 1 ROCm device";
        }

        // 1MB = 262144 floats (1048576 bytes)
        const size_t count = 262144;
        const size_t bytes = count * sizeof(float);

        // Allocate buffers
        void *cuda_buf = allocateCUDA(bytes, 0);
        void *rocm_buf = allocateROCm(bytes, 0);

        ASSERT_NE(cuda_buf, nullptr) << "Failed to allocate CUDA buffer";
        ASSERT_NE(rocm_buf, nullptr) << "Failed to allocate ROCm buffer";

        // Initialize with simple pattern
        std::vector<float> cuda_data(count);
        std::vector<float> rocm_data(count);
        for (size_t i = 0; i < count; ++i)
        {
            cuda_data[i] = 1.0f; // All 1s
            rocm_data[i] = 2.0f; // All 2s
        }

        ASSERT_TRUE(hostToDeviceCUDA(cuda_buf, cuda_data.data(), bytes, 0));
        ASSERT_TRUE(hostToDeviceROCm(rocm_buf, rocm_data.data(), bytes, 0));

        // Synchronize transfers
        cuda_backend_->synchronize(0);
        rocm_backend_->synchronize(0);

        // Create and initialize HeterogeneousBackend
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

        // Execute Phase 2 (bridge exchange)
        ASSERT_TRUE(backend.executePhase2_BridgeExchange(
            cuda_buf, rocm_buf, count,
            CollectiveDataType::FLOAT32,
            CollectiveOp::ALLREDUCE_SUM));

        // Synchronize
        ASSERT_TRUE(backend.synchronize());

        // Read back results (sample first and last few elements)
        std::vector<float> cuda_result(count);
        std::vector<float> rocm_result(count);

        ASSERT_TRUE(deviceToHostCUDA(cuda_result.data(), cuda_buf, bytes, 0));
        ASSERT_TRUE(deviceToHostROCm(rocm_result.data(), rocm_buf, bytes, 0));

        // Expected: all elements should be 3.0f (1.0f + 2.0f)
        const float expected = 3.0f;

        // Check first 10 elements
        for (size_t i = 0; i < 10; ++i)
        {
            EXPECT_FLOAT_EQ(cuda_result[i], expected)
                << "CUDA result mismatch at index " << i;
            EXPECT_FLOAT_EQ(rocm_result[i], expected)
                << "ROCm result mismatch at index " << i;
        }

        // Check last 10 elements
        for (size_t i = count - 10; i < count; ++i)
        {
            EXPECT_FLOAT_EQ(cuda_result[i], expected)
                << "CUDA result mismatch at index " << i;
            EXPECT_FLOAT_EQ(rocm_result[i], expected)
                << "ROCm result mismatch at index " << i;
        }

        // Check middle element
        EXPECT_FLOAT_EQ(cuda_result[count / 2], expected);
        EXPECT_FLOAT_EQ(rocm_result[count / 2], expected);

        backend.shutdown();
    }

    TEST_F(Test__HeterogeneousBackend_Phase2, Phase2FailsWhenNotInitialized)
    {
        // executePhase2 should fail if backend is not initialized
        HeterogeneousBackend backend;

        void *dummy_cuda = reinterpret_cast<void *>(0x1000);
        void *dummy_rocm = reinterpret_cast<void *>(0x2000);

        EXPECT_FALSE(backend.executePhase2_BridgeExchange(
            dummy_cuda, dummy_rocm, 4,
            CollectiveDataType::FLOAT32,
            CollectiveOp::ALLREDUCE_SUM));
    }

#else

    TEST_F(Test__HeterogeneousBackend_Phase2, SkipWhenNoCUDAOrROCm)
    {
        GTEST_SKIP() << "HeterogeneousBackend Phase 2 tests require both HAVE_CUDA and HAVE_ROCM";
    }

#endif // defined(HAVE_CUDA) && defined(HAVE_ROCM)

} // namespace llaminar2::test
