/**
 * @file Test__HeterogeneousBackend_FullAllReduce.cpp
 * @brief Integration tests for HeterogeneousBackend full 3-phase allreduce
 *
 * These tests require actual GPU hardware and test the complete allreduce:
 * - Phase 1: Intra-domain reduce (NCCL for CUDA, RCCL for ROCm)
 * - Phase 2: Bridge exchange via HOST staging
 * - Phase 3: Intra-domain broadcast
 *
 * Test naming: V2_Integration_HeterogeneousBackend_FullAllReduce_*
 *
 * @author David Sanftenberg
 * @date January 2026
 */

#include <gtest/gtest.h>
#include "v2/collective/backends/HeterogeneousBackend.h"
#include "v2/collective/LocalTPContext.h"
#include "v2/collective/DeviceGroup.h"
#include "v2/backends/DeviceId.h"
#include "v2/backends/GlobalDeviceAddress.h"
#include "v2/backends/BackendManager.h"
#include "v2/backends/IBackend.h"
#include "v2/config/OrchestrationConfig.h"

#include <cstring>
#include <numeric>

namespace llaminar2::test
{

    // ═══════════════════════════════════════════════════════════════════════════
    // Test Fixture
    // ═══════════════════════════════════════════════════════════════════════════

    class Test__HeterogeneousBackend_FullAllReduce : public ::testing::Test
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
            // Free any allocated buffers
            for (auto &[buf, info] : allocated_buffers_)
            {
                if (info.type == DeviceType::CUDA && cuda_backend_)
                {
                    cuda_backend_->free(buf, info.device_idx);
                }
                else if (info.type == DeviceType::ROCm && rocm_backend_)
                {
                    rocm_backend_->free(buf, info.device_idx);
                }
            }
            allocated_buffers_.clear();
        }

        // ─────────────────────────────────────────────────────────────────────
        // Buffer Management
        // ─────────────────────────────────────────────────────────────────────

        struct BufferInfo
        {
            DeviceType type;
            int device_idx;
        };

        void *allocateBuffer(DeviceType type, int device_idx, size_t bytes)
        {
            void *buf = nullptr;
            if (type == DeviceType::CUDA && cuda_backend_)
            {
                buf = cuda_backend_->allocate(bytes, device_idx);
            }
            else if (type == DeviceType::ROCm && rocm_backend_)
            {
                buf = rocm_backend_->allocate(bytes, device_idx);
            }

            if (buf)
            {
                allocated_buffers_[buf] = {type, device_idx};
            }
            return buf;
        }

        bool initializeBuffer(void *buf, const float *data, size_t bytes, DeviceType type, int device_idx)
        {
            if (type == DeviceType::CUDA && cuda_backend_)
            {
                return cuda_backend_->hostToDevice(buf, data, bytes, device_idx);
            }
            else if (type == DeviceType::ROCm && rocm_backend_)
            {
                return rocm_backend_->hostToDevice(buf, data, bytes, device_idx);
            }
            return false;
        }

        bool readBuffer(void *buf, float *out, size_t bytes, DeviceType type, int device_idx)
        {
            if (type == DeviceType::CUDA && cuda_backend_)
            {
                return cuda_backend_->deviceToHost(out, buf, bytes, device_idx);
            }
            else if (type == DeviceType::ROCm && rocm_backend_)
            {
                return rocm_backend_->deviceToHost(out, buf, bytes, device_idx);
            }
            return false;
        }

        IBackend *cuda_backend_ = nullptr;
        IBackend *rocm_backend_ = nullptr;
        int cuda_count_ = 0;
        int rocm_count_ = 0;
        std::map<void *, BufferInfo> allocated_buffers_;
    };

    // ═══════════════════════════════════════════════════════════════════════════
    // Full AllReduce Integration Tests
    // ═══════════════════════════════════════════════════════════════════════════

#if defined(HAVE_CUDA) && defined(HAVE_ROCM)

    /**
     * @test Full 3-phase allreduce with 1 CUDA + 2 ROCm devices
     *
     * Configuration:
     * - cuda:0 = [1, 1, 1]
     * - rocm:0 = [2, 2, 2]
     * - rocm:1 = [3, 3, 3]
     *
     * Expected result (global sum): ALL buffers = [6, 6, 6]
     */
    TEST_F(Test__HeterogeneousBackend_FullAllReduce, AllReduce_1Cuda2Rocm)
    {
        // Skip if hardware requirements not met
        if (cuda_count_ < 1)
        {
            GTEST_SKIP() << "Test requires at least 1 CUDA device, found " << cuda_count_;
        }
        if (rocm_count_ < 2)
        {
            GTEST_SKIP() << "Test requires at least 2 ROCm devices, found " << rocm_count_;
        }

        // TODO: Remove skip once RCCLBackend::reduceMulti() is implemented in coordinator mode
        GTEST_SKIP() << "RCCL multi-device collective operations not yet implemented in coordinator mode";

        const size_t count = 3;
        const size_t bytes = count * sizeof(float);

        // Allocate buffers
        void *cuda_buf_0 = allocateBuffer(DeviceType::CUDA, 0, bytes);
        void *rocm_buf_0 = allocateBuffer(DeviceType::ROCm, 0, bytes);
        void *rocm_buf_1 = allocateBuffer(DeviceType::ROCm, 1, bytes);

        ASSERT_NE(cuda_buf_0, nullptr) << "Failed to allocate CUDA buffer on device 0";
        ASSERT_NE(rocm_buf_0, nullptr) << "Failed to allocate ROCm buffer on device 0";
        ASSERT_NE(rocm_buf_1, nullptr) << "Failed to allocate ROCm buffer on device 1";

        // Initialize buffers with known values
        float cuda_data[] = {1.0f, 1.0f, 1.0f};
        float rocm_data_0[] = {2.0f, 2.0f, 2.0f};
        float rocm_data_1[] = {3.0f, 3.0f, 3.0f};

        ASSERT_TRUE(initializeBuffer(cuda_buf_0, cuda_data, bytes, DeviceType::CUDA, 0));
        ASSERT_TRUE(initializeBuffer(rocm_buf_0, rocm_data_0, bytes, DeviceType::ROCm, 0));
        ASSERT_TRUE(initializeBuffer(rocm_buf_1, rocm_data_1, bytes, DeviceType::ROCm, 1));

        // Create HeterogeneousBackend and initialize with device group
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

        ASSERT_TRUE(backend.initialize(group)) << "Failed to initialize HeterogeneousBackend";
        ASSERT_TRUE(backend.isInitialized());

        // Verify it's a multi-GPU backend
        EXPECT_TRUE(backend.isMultiGpuSingleProcess());

        // Execute allreduceMulti
        // Buffer order matches device order: [cuda:0, rocm:0, rocm:1]
        std::vector<void *> buffers = {cuda_buf_0, rocm_buf_0, rocm_buf_1};

        ASSERT_TRUE(backend.allreduceMulti(buffers, count, CollectiveDataType::FLOAT32,
                                           CollectiveOp::ALLREDUCE_SUM))
            << "allreduceMulti failed: " << backend.lastError();

        // Synchronize to ensure all operations complete
        ASSERT_TRUE(backend.synchronize());

        // Read back and verify ALL buffers have the global sum [6, 6, 6]
        float expected[] = {6.0f, 6.0f, 6.0f};

        float cuda_result[3] = {0};
        float rocm_result_0[3] = {0};
        float rocm_result_1[3] = {0};

        ASSERT_TRUE(readBuffer(cuda_buf_0, cuda_result, bytes, DeviceType::CUDA, 0));
        ASSERT_TRUE(readBuffer(rocm_buf_0, rocm_result_0, bytes, DeviceType::ROCm, 0));
        ASSERT_TRUE(readBuffer(rocm_buf_1, rocm_result_1, bytes, DeviceType::ROCm, 1));

        // Verify cuda:0 result
        for (size_t i = 0; i < count; ++i)
        {
            EXPECT_FLOAT_EQ(cuda_result[i], expected[i])
                << "cuda:0 result mismatch at index " << i;
        }

        // Verify rocm:0 result
        for (size_t i = 0; i < count; ++i)
        {
            EXPECT_FLOAT_EQ(rocm_result_0[i], expected[i])
                << "rocm:0 result mismatch at index " << i;
        }

        // Verify rocm:1 result
        for (size_t i = 0; i < count; ++i)
        {
            EXPECT_FLOAT_EQ(rocm_result_1[i], expected[i])
                << "rocm:1 result mismatch at index " << i;
        }

        backend.shutdown();
    }

    /**
     * @test Full allreduce via LocalTPContext with HETEROGENEOUS backend
     *
     * Verifies that LocalTPContext correctly selects and uses HeterogeneousBackend
     * for mixed CUDA+ROCm configurations with >2 devices.
     */
    TEST_F(Test__HeterogeneousBackend_FullAllReduce, ViaLocalTPContext_1Cuda2Rocm)
    {
        // Skip if hardware requirements not met
        if (cuda_count_ < 1)
        {
            GTEST_SKIP() << "Test requires at least 1 CUDA device, found " << cuda_count_;
        }
        if (rocm_count_ < 2)
        {
            GTEST_SKIP() << "Test requires at least 2 ROCm devices, found " << rocm_count_;
        }

        const size_t count = 4;
        const size_t bytes = count * sizeof(float);

        // Allocate buffers
        void *cuda_buf_0 = allocateBuffer(DeviceType::CUDA, 0, bytes);
        void *rocm_buf_0 = allocateBuffer(DeviceType::ROCm, 0, bytes);
        void *rocm_buf_1 = allocateBuffer(DeviceType::ROCm, 1, bytes);

        ASSERT_NE(cuda_buf_0, nullptr);
        ASSERT_NE(rocm_buf_0, nullptr);
        ASSERT_NE(rocm_buf_1, nullptr);

        // Initialize with different values
        float cuda_data[] = {10.0f, 20.0f, 30.0f, 40.0f};
        float rocm_data_0[] = {1.0f, 2.0f, 3.0f, 4.0f};
        float rocm_data_1[] = {100.0f, 200.0f, 300.0f, 400.0f};

        ASSERT_TRUE(initializeBuffer(cuda_buf_0, cuda_data, bytes, DeviceType::CUDA, 0));
        ASSERT_TRUE(initializeBuffer(rocm_buf_0, rocm_data_0, bytes, DeviceType::ROCm, 0));
        ASSERT_TRUE(initializeBuffer(rocm_buf_1, rocm_data_1, bytes, DeviceType::ROCm, 1));

        // Create LocalTPContext with AUTO backend selection
        // Should automatically select HETEROGENEOUS for 1 CUDA + 2 ROCm
        std::vector<GlobalDeviceAddress> devices = {
            GlobalDeviceAddress::cuda(0, 0),
            GlobalDeviceAddress::rocm(0, 0),
            GlobalDeviceAddress::rocm(1, 0)};

        auto ctx = createLocalTPContext(devices, {}, CollectiveBackendType::AUTO);
        ASSERT_NE(ctx, nullptr);

        // Verify it selected HETEROGENEOUS backend
        EXPECT_EQ(ctx->backend(), CollectiveBackendType::HETEROGENEOUS)
            << "Expected HETEROGENEOUS backend for 1 CUDA + 2 ROCm";

        // Note: LocalTPContext::allreduce() works with TensorBase objects,
        // not raw buffers. For raw buffer testing, we use HeterogeneousBackend
        // directly as shown in the previous test.
        //
        // This test primarily verifies that LocalTPContext correctly selects
        // the HETEROGENEOUS backend for >2 mixed devices.
    }

    /**
     * @test Full 3-phase allreduce with 2 CUDA + 1 ROCm devices
     */
    TEST_F(Test__HeterogeneousBackend_FullAllReduce, AllReduce_2Cuda1Rocm)
    {
        // Skip if hardware requirements not met
        if (cuda_count_ < 2)
        {
            GTEST_SKIP() << "Test requires at least 2 CUDA devices, found " << cuda_count_;
        }
        if (rocm_count_ < 1)
        {
            GTEST_SKIP() << "Test requires at least 1 ROCm device, found " << rocm_count_;
        }

        // TODO: Remove skip once NCCLCoordinator::reduceMulti() is fully implemented
        GTEST_SKIP() << "NCCL multi-device collective operations not yet fully implemented in coordinator mode";

        const size_t count = 3;
        const size_t bytes = count * sizeof(float);

        // Allocate buffers
        void *cuda_buf_0 = allocateBuffer(DeviceType::CUDA, 0, bytes);
        void *cuda_buf_1 = allocateBuffer(DeviceType::CUDA, 1, bytes);
        void *rocm_buf_0 = allocateBuffer(DeviceType::ROCm, 0, bytes);

        ASSERT_NE(cuda_buf_0, nullptr);
        ASSERT_NE(cuda_buf_1, nullptr);
        ASSERT_NE(rocm_buf_0, nullptr);

        // Initialize: cuda:0=[1,1,1], cuda:1=[2,2,2], rocm:0=[3,3,3]
        // Expected sum: [6,6,6]
        float cuda_data_0[] = {1.0f, 1.0f, 1.0f};
        float cuda_data_1[] = {2.0f, 2.0f, 2.0f};
        float rocm_data_0[] = {3.0f, 3.0f, 3.0f};

        ASSERT_TRUE(initializeBuffer(cuda_buf_0, cuda_data_0, bytes, DeviceType::CUDA, 0));
        ASSERT_TRUE(initializeBuffer(cuda_buf_1, cuda_data_1, bytes, DeviceType::CUDA, 1));
        ASSERT_TRUE(initializeBuffer(rocm_buf_0, rocm_data_0, bytes, DeviceType::ROCm, 0));

        // Create and initialize backend
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

        // Execute allreduceMulti
        std::vector<void *> buffers = {cuda_buf_0, cuda_buf_1, rocm_buf_0};

        ASSERT_TRUE(backend.allreduceMulti(buffers, count, CollectiveDataType::FLOAT32,
                                           CollectiveOp::ALLREDUCE_SUM));
        ASSERT_TRUE(backend.synchronize());

        // Verify all buffers have [6, 6, 6]
        float expected[] = {6.0f, 6.0f, 6.0f};

        float cuda_result_0[3], cuda_result_1[3], rocm_result_0[3];

        ASSERT_TRUE(readBuffer(cuda_buf_0, cuda_result_0, bytes, DeviceType::CUDA, 0));
        ASSERT_TRUE(readBuffer(cuda_buf_1, cuda_result_1, bytes, DeviceType::CUDA, 1));
        ASSERT_TRUE(readBuffer(rocm_buf_0, rocm_result_0, bytes, DeviceType::ROCm, 0));

        for (size_t i = 0; i < count; ++i)
        {
            EXPECT_FLOAT_EQ(cuda_result_0[i], expected[i]) << "cuda:0 mismatch at " << i;
            EXPECT_FLOAT_EQ(cuda_result_1[i], expected[i]) << "cuda:1 mismatch at " << i;
            EXPECT_FLOAT_EQ(rocm_result_0[i], expected[i]) << "rocm:0 mismatch at " << i;
        }

        backend.shutdown();
    }

    /**
     * @test Full 3-phase allreduce with 2 CUDA + 2 ROCm devices (4-way TP)
     */
    TEST_F(Test__HeterogeneousBackend_FullAllReduce, AllReduce_2Cuda2Rocm)
    {
        // Skip if hardware requirements not met
        if (cuda_count_ < 2)
        {
            GTEST_SKIP() << "Test requires at least 2 CUDA devices, found " << cuda_count_;
        }
        if (rocm_count_ < 2)
        {
            GTEST_SKIP() << "Test requires at least 2 ROCm devices, found " << rocm_count_;
        }

        // TODO: Remove skip once NCCL/RCCL coordinator multi-device operations are fully implemented
        GTEST_SKIP() << "Multi-device collective operations not yet fully implemented in coordinator mode";

        const size_t count = 4;
        const size_t bytes = count * sizeof(float);

        // Allocate buffers
        void *cuda_buf_0 = allocateBuffer(DeviceType::CUDA, 0, bytes);
        void *cuda_buf_1 = allocateBuffer(DeviceType::CUDA, 1, bytes);
        void *rocm_buf_0 = allocateBuffer(DeviceType::ROCm, 0, bytes);
        void *rocm_buf_1 = allocateBuffer(DeviceType::ROCm, 1, bytes);

        ASSERT_NE(cuda_buf_0, nullptr);
        ASSERT_NE(cuda_buf_1, nullptr);
        ASSERT_NE(rocm_buf_0, nullptr);
        ASSERT_NE(rocm_buf_1, nullptr);

        // Initialize: each device has value = device_index + 1
        // cuda:0=[1,1,1,1], cuda:1=[2,2,2,2], rocm:0=[3,3,3,3], rocm:1=[4,4,4,4]
        // Expected sum: [10,10,10,10]
        float cuda_data_0[] = {1.0f, 1.0f, 1.0f, 1.0f};
        float cuda_data_1[] = {2.0f, 2.0f, 2.0f, 2.0f};
        float rocm_data_0[] = {3.0f, 3.0f, 3.0f, 3.0f};
        float rocm_data_1[] = {4.0f, 4.0f, 4.0f, 4.0f};

        ASSERT_TRUE(initializeBuffer(cuda_buf_0, cuda_data_0, bytes, DeviceType::CUDA, 0));
        ASSERT_TRUE(initializeBuffer(cuda_buf_1, cuda_data_1, bytes, DeviceType::CUDA, 1));
        ASSERT_TRUE(initializeBuffer(rocm_buf_0, rocm_data_0, bytes, DeviceType::ROCm, 0));
        ASSERT_TRUE(initializeBuffer(rocm_buf_1, rocm_data_1, bytes, DeviceType::ROCm, 1));

        // Create and initialize backend
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

        // Execute allreduceMulti
        std::vector<void *> buffers = {cuda_buf_0, cuda_buf_1, rocm_buf_0, rocm_buf_1};

        ASSERT_TRUE(backend.allreduceMulti(buffers, count, CollectiveDataType::FLOAT32,
                                           CollectiveOp::ALLREDUCE_SUM));
        ASSERT_TRUE(backend.synchronize());

        // Verify all buffers have [10, 10, 10, 10]
        float expected[] = {10.0f, 10.0f, 10.0f, 10.0f};

        float cuda_result_0[4], cuda_result_1[4], rocm_result_0[4], rocm_result_1[4];

        ASSERT_TRUE(readBuffer(cuda_buf_0, cuda_result_0, bytes, DeviceType::CUDA, 0));
        ASSERT_TRUE(readBuffer(cuda_buf_1, cuda_result_1, bytes, DeviceType::CUDA, 1));
        ASSERT_TRUE(readBuffer(rocm_buf_0, rocm_result_0, bytes, DeviceType::ROCm, 0));
        ASSERT_TRUE(readBuffer(rocm_buf_1, rocm_result_1, bytes, DeviceType::ROCm, 1));

        for (size_t i = 0; i < count; ++i)
        {
            EXPECT_FLOAT_EQ(cuda_result_0[i], expected[i]) << "cuda:0 mismatch at " << i;
            EXPECT_FLOAT_EQ(cuda_result_1[i], expected[i]) << "cuda:1 mismatch at " << i;
            EXPECT_FLOAT_EQ(rocm_result_0[i], expected[i]) << "rocm:0 mismatch at " << i;
            EXPECT_FLOAT_EQ(rocm_result_1[i], expected[i]) << "rocm:1 mismatch at " << i;
        }

        backend.shutdown();
    }

#endif // defined(HAVE_CUDA) && defined(HAVE_ROCM)

} // namespace llaminar2::test
