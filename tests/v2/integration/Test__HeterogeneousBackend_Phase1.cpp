/**
 * @file Test__HeterogeneousBackend_Phase1.cpp
 * @brief Integration tests for HeterogeneousBackend Phase 1 (Intra-Domain Reduce)
 *
 * These tests require actual GPU hardware and test the Phase 1 algorithm:
 * - NCCL reduce across CUDA devices (if >1 CUDA)
 * - RCCL reduce across ROCm devices (if >1 ROCm)
 *
 * NOTE: This test uses global backend accessors for device allocation to avoid
 * directly including both cuda_runtime.h and hip_runtime.h (which conflict).
 *
 * Test naming: V2_Integration_HeterogeneousBackend_Phase1_*
 *
 * @author David Sanftenberg
 * @date January 2026
 */

#include <gtest/gtest.h>
#include "v2/collective/backends/HeterogeneousBackend.h"
#include "v2/collective/backends/NCCLBackend.h"
#include "v2/collective/backends/RCCLBackend.h"
#include "v2/collective/DeviceGroup.h"
#include "v2/backends/DeviceId.h"
#include "v2/backends/BackendManager.h"
#include "v2/backends/IBackend.h"

namespace llaminar2::test
{

    // ═══════════════════════════════════════════════════════════════════════════
    // Test Fixture
    // ═══════════════════════════════════════════════════════════════════════════

    class Test__HeterogeneousBackend_Phase1 : public ::testing::Test
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

        IBackend *cuda_backend_ = nullptr;
        IBackend *rocm_backend_ = nullptr;
        int cuda_count_ = 0;
        int rocm_count_ = 0;
    };

    // ═══════════════════════════════════════════════════════════════════════════
    // Phase 1 Integration Tests
    // ═══════════════════════════════════════════════════════════════════════════

#if defined(HAVE_CUDA) && defined(HAVE_ROCM)

    TEST_F(Test__HeterogeneousBackend_Phase1, RCCLReduceOn2AMD)
    {
        // Skip if < 2 ROCm devices available
        if (rocm_count_ < 2)
        {
            GTEST_SKIP() << "Test requires at least 2 ROCm devices, found " << rocm_count_;
        }

        // Also need at least 1 CUDA device for HeterogeneousBackend
        if (cuda_count_ < 1)
        {
            GTEST_SKIP() << "Test requires at least 1 CUDA device, found " << cuda_count_;
        }

        const size_t count = 3;
        const size_t bytes = count * sizeof(float);

        // Allocate buffers using IBackend interface (avoids header conflicts)
        ASSERT_NE(rocm_backend_, nullptr) << "ROCm backend not available";
        ASSERT_NE(cuda_backend_, nullptr) << "CUDA backend not available";

        void *rocm_buf_0 = rocm_backend_->allocate(bytes, 0);
        void *rocm_buf_1 = rocm_backend_->allocate(bytes, 1);
        void *cuda_buf_0 = cuda_backend_->allocate(bytes, 0);

        ASSERT_NE(rocm_buf_0, nullptr) << "Failed to allocate ROCm buffer on device 0";
        ASSERT_NE(rocm_buf_1, nullptr) << "Failed to allocate ROCm buffer on device 1";
        ASSERT_NE(cuda_buf_0, nullptr) << "Failed to allocate CUDA buffer on device 0";

        // Initialize buffers with known values
        // rocm:0 = [1, 2, 3], rocm:1 = [4, 5, 6]
        // Expected sum: [5, 7, 9]
        float rocm_data_0[] = {1.0f, 2.0f, 3.0f};
        float rocm_data_1[] = {4.0f, 5.0f, 6.0f};
        float cuda_data_0[] = {10.0f, 20.0f, 30.0f};

        ASSERT_TRUE(rocm_backend_->hostToDevice(rocm_buf_0, rocm_data_0, bytes, 0));
        ASSERT_TRUE(rocm_backend_->hostToDevice(rocm_buf_1, rocm_data_1, bytes, 1));
        ASSERT_TRUE(cuda_backend_->hostToDevice(cuda_buf_0, cuda_data_0, bytes, 0));

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
        ASSERT_TRUE(backend.isInitialized());

        // Verify the plan
        auto plan = backend.planPhase1();
        EXPECT_FALSE(plan.will_call_nccl_reduce); // Only 1 CUDA device
        EXPECT_TRUE(plan.will_call_rccl_reduce);  // 2 ROCm devices
        EXPECT_EQ(plan.nccl_reduce_root, -1);
        EXPECT_EQ(plan.rccl_reduce_root, 0); // rocm:0 is the bridge

        // Get the RCCL backend (created during HeterogeneousBackend::initialize)
        ASSERT_TRUE(backend.hasRCCLBackend());

        // Synchronize after reduce
        ASSERT_TRUE(backend.synchronize());

        // Cleanup
        backend.shutdown();

        rocm_backend_->free(rocm_buf_0, 0);
        rocm_backend_->free(rocm_buf_1, 1);
        cuda_backend_->free(cuda_buf_0, 0);
    }

    TEST_F(Test__HeterogeneousBackend_Phase1, SkipWithSingleDevicePerDomain)
    {
        // With 1 CUDA + 1 ROCm, Phase 1 should be a no-op (no reduce needed)
        if (cuda_count_ < 1 || rocm_count_ < 1)
        {
            GTEST_SKIP() << "Test requires at least 1 CUDA and 1 ROCm device";
        }

        // Create backend with minimal config
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
        ASSERT_TRUE(backend.isInitialized());

        // Verify the plan shows no reduces will be called
        auto plan = backend.planPhase1();
        EXPECT_FALSE(plan.will_call_nccl_reduce);
        EXPECT_FALSE(plan.will_call_rccl_reduce);
        EXPECT_EQ(plan.nccl_device_count, 1u);
        EXPECT_EQ(plan.rccl_device_count, 1u);

        // No NCCL or RCCL backend should be created for single-device domains
        EXPECT_FALSE(backend.hasNCCLBackend());
        EXPECT_FALSE(backend.hasRCCLBackend());

        // Bridge backend should still be created
        EXPECT_TRUE(backend.hasBridgeBackend());

        backend.shutdown();
    }

    TEST_F(Test__HeterogeneousBackend_Phase1, PlanCorrectFor2Cuda2Rocm)
    {
        // With 2 CUDA + 2 ROCm, both NCCL and RCCL reduces should be called
        if (cuda_count_ < 2 || rocm_count_ < 2)
        {
            GTEST_SKIP() << "Test requires at least 2 CUDA and 2 ROCm devices";
        }

        HeterogeneousBackend backend;

        DeviceGroupBuilder builder;
        auto group = builder
                         .setName("2_cuda_2_rocm")
                         .setScope(CollectiveScope::LOCAL)
                         .addDevice(DeviceId::cuda(0))
                         .addDevice(DeviceId::cuda(1))
                         .addDevice(DeviceId::rocm(0))
                         .addDevice(DeviceId::rocm(1))
                         .setLocalRank(0)
                         .build();

        ASSERT_TRUE(backend.initialize(group));

        auto plan = backend.planPhase1();
        EXPECT_TRUE(plan.will_call_nccl_reduce);
        EXPECT_TRUE(plan.will_call_rccl_reduce);
        EXPECT_EQ(plan.nccl_reduce_root, 0); // cuda:0 is bridge
        EXPECT_EQ(plan.rccl_reduce_root, 0); // rocm:0 is bridge
        EXPECT_EQ(plan.nccl_device_count, 2u);
        EXPECT_EQ(plan.rccl_device_count, 2u);

        EXPECT_TRUE(backend.hasNCCLBackend());
        EXPECT_TRUE(backend.hasRCCLBackend());
        EXPECT_TRUE(backend.hasBridgeBackend());

        backend.shutdown();
    }

    TEST_F(Test__HeterogeneousBackend_Phase1, PlanCorrectFor2Cuda1Rocm)
    {
        // With 2 CUDA + 1 ROCm, only NCCL reduce should be called
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

        auto plan = backend.planPhase1();
        EXPECT_TRUE(plan.will_call_nccl_reduce);  // 2 CUDA devices
        EXPECT_FALSE(plan.will_call_rccl_reduce); // Only 1 ROCm device
        EXPECT_EQ(plan.nccl_reduce_root, 0);
        EXPECT_EQ(plan.rccl_reduce_root, -1);
        EXPECT_EQ(plan.nccl_device_count, 2u);
        EXPECT_EQ(plan.rccl_device_count, 1u);

        EXPECT_TRUE(backend.hasNCCLBackend());
        EXPECT_FALSE(backend.hasRCCLBackend());

        backend.shutdown();
    }

#else

    TEST_F(Test__HeterogeneousBackend_Phase1, SkipWhenNoCUDAOrROCm)
    {
        GTEST_SKIP() << "HeterogeneousBackend tests require both HAVE_CUDA and HAVE_ROCM";
    }

#endif // defined(HAVE_CUDA) && defined(HAVE_ROCM)

} // namespace llaminar2::test
