/**
 * @file Test__HostBackend.cpp
 * @brief Unit tests for HostBackend
 *
 * Tests the CPU-based fallback collective backend.
 */

#include <gtest/gtest.h>
#include "v2/collective/backends/HostBackend.h"
#include "v2/collective/DeviceGroup.h"
#include "v2/backends/DeviceId.h"

namespace llaminar2::test
{

    // ═══════════════════════════════════════════════════════════════════════════
    // Test Fixture
    // ═══════════════════════════════════════════════════════════════════════════

    class Test__HostBackend : public ::testing::Test
    {
    protected:
        void SetUp() override
        {
            backend_ = std::make_unique<HostBackend>();
        }

        void TearDown() override
        {
            if (backend_ && backend_->isInitialized())
            {
                backend_->shutdown();
            }
        }

        // Helper to create a single-device CPU group
        DeviceGroup createSingleCPUGroup()
        {
            DeviceGroupBuilder builder;
            return builder
                .setName("single_cpu")
                .setScope(CollectiveScope::LOCAL)
                .addDevice(DeviceId::cpu())
                .setLocalRank(0)
                .build();
        }

        // Helper to create a single-device CUDA group
        DeviceGroup createSingleCUDAGroup()
        {
            DeviceGroupBuilder builder;
            return builder
                .setName("single_cuda")
                .setScope(CollectiveScope::LOCAL)
                .addDevice(DeviceId::cuda(0))
                .setLocalRank(0)
                .build();
        }

        // Helper to create a multi-device CUDA group
        DeviceGroup createMultiCUDAGroup()
        {
            DeviceGroupBuilder builder;
            return builder
                .setName("multi_cuda")
                .setScope(CollectiveScope::LOCAL)
                .addDevice(DeviceId::cuda(0))
                .addDevice(DeviceId::cuda(1))
                .setLocalRank(0)
                .build();
        }

        // Helper to create a heterogeneous group
        DeviceGroup createHeterogeneousGroup()
        {
            DeviceGroupBuilder builder;
            return builder
                .setName("hetero")
                .setScope(CollectiveScope::LOCAL)
                .addDevice(DeviceId::cuda(0))
                .addDevice(DeviceId::cpu())
                .setLocalRank(0)
                .build();
        }

        std::unique_ptr<HostBackend> backend_;
    };

    // ═══════════════════════════════════════════════════════════════════════════
    // Identity Tests
    // ═══════════════════════════════════════════════════════════════════════════

    TEST_F(Test__HostBackend, TypeReturnsHost)
    {
        EXPECT_EQ(backend_->type(), CollectiveBackendType::HOST);
    }

    TEST_F(Test__HostBackend, NameReturnsHost)
    {
        EXPECT_EQ(backend_->name(), "Host");
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // Availability Tests
    // ═══════════════════════════════════════════════════════════════════════════

    TEST_F(Test__HostBackend, IsAlwaysAvailable)
    {
        EXPECT_TRUE(backend_->isAvailable());
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // Device Support Tests
    // ═══════════════════════════════════════════════════════════════════════════

    TEST_F(Test__HostBackend, SupportsAllDeviceTypes)
    {
        // HostBackend supports all device types via host memory staging
        EXPECT_TRUE(backend_->supportsDevice(DeviceType::CPU));
        EXPECT_TRUE(backend_->supportsDevice(DeviceType::CUDA));
        EXPECT_TRUE(backend_->supportsDevice(DeviceType::ROCm));
    }

    TEST_F(Test__HostBackend, SupportsDirectTransferOnlyForCPU)
    {
        // CPU ↔ CPU: Direct transfer possible
        EXPECT_TRUE(backend_->supportsDirectTransfer(DeviceId::cpu(), DeviceId::cpu()));

        // GPU ↔ GPU: Requires host staging
        EXPECT_FALSE(backend_->supportsDirectTransfer(DeviceId::cuda(0), DeviceId::cuda(1)));
        EXPECT_FALSE(backend_->supportsDirectTransfer(DeviceId::rocm(0), DeviceId::rocm(1)));

        // CPU ↔ GPU: Requires host staging
        EXPECT_FALSE(backend_->supportsDirectTransfer(DeviceId::cpu(), DeviceId::cuda(0)));
        EXPECT_FALSE(backend_->supportsDirectTransfer(DeviceId::cuda(0), DeviceId::cpu()));

        // CUDA ↔ ROCm: Requires host staging
        EXPECT_FALSE(backend_->supportsDirectTransfer(DeviceId::cuda(0), DeviceId::rocm(0)));
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // Lifecycle Tests
    // ═══════════════════════════════════════════════════════════════════════════

    TEST_F(Test__HostBackend, InitializeSucceeds)
    {
        auto group = createSingleCPUGroup();
        EXPECT_TRUE(backend_->initialize(group));
    }

    TEST_F(Test__HostBackend, IsInitializedAfterInit)
    {
        EXPECT_FALSE(backend_->isInitialized());

        auto group = createSingleCPUGroup();
        backend_->initialize(group);

        EXPECT_TRUE(backend_->isInitialized());
    }

    TEST_F(Test__HostBackend, ShutdownClearsInitialized)
    {
        auto group = createSingleCPUGroup();
        backend_->initialize(group);
        EXPECT_TRUE(backend_->isInitialized());

        backend_->shutdown();
        EXPECT_FALSE(backend_->isInitialized());
    }

    TEST_F(Test__HostBackend, CanReinitializeAfterShutdown)
    {
        auto group1 = createSingleCPUGroup();
        backend_->initialize(group1);
        backend_->shutdown();

        auto group2 = createSingleCUDAGroup();
        EXPECT_TRUE(backend_->initialize(group2));
        EXPECT_TRUE(backend_->isInitialized());
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // AllReduce Tests
    // ═══════════════════════════════════════════════════════════════════════════

    TEST_F(Test__HostBackend, AllreduceSucceedsWhenInitialized)
    {
        auto group = createSingleCPUGroup();
        backend_->initialize(group);

        std::vector<float> buffer = {1.0f, 2.0f, 3.0f, 4.0f};
        EXPECT_TRUE(backend_->allreduce(
            buffer.data(),
            buffer.size(),
            CollectiveDataType::FLOAT32,
            CollectiveOp::ALLREDUCE_SUM));
    }

    TEST_F(Test__HostBackend, AllreduceFailsWhenNotInitialized)
    {
        std::vector<float> buffer = {1.0f, 2.0f, 3.0f, 4.0f};
        EXPECT_FALSE(backend_->allreduce(
            buffer.data(),
            buffer.size(),
            CollectiveDataType::FLOAT32,
            CollectiveOp::ALLREDUCE_SUM));
    }

    TEST_F(Test__HostBackend, AllreducePreservesDataForSingleDevice)
    {
        // For single-device groups, allreduce is a no-op
        auto group = createSingleCPUGroup();
        backend_->initialize(group);

        std::vector<float> buffer = {1.0f, 2.0f, 3.0f, 4.0f};
        std::vector<float> expected = buffer;

        backend_->allreduce(
            buffer.data(),
            buffer.size(),
            CollectiveDataType::FLOAT32,
            CollectiveOp::ALLREDUCE_SUM);

        EXPECT_EQ(buffer, expected);
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // AllGather Tests
    // ═══════════════════════════════════════════════════════════════════════════

    TEST_F(Test__HostBackend, AllgatherSucceedsWhenInitialized)
    {
        auto group = createSingleCPUGroup();
        backend_->initialize(group);

        std::vector<float> send_buf = {1.0f, 2.0f, 3.0f, 4.0f};
        std::vector<float> recv_buf(send_buf.size(), 0.0f);

        EXPECT_TRUE(backend_->allgather(
            send_buf.data(),
            recv_buf.data(),
            send_buf.size(),
            CollectiveDataType::FLOAT32));
    }

    TEST_F(Test__HostBackend, AllgatherFailsWhenNotInitialized)
    {
        std::vector<float> send_buf = {1.0f, 2.0f, 3.0f, 4.0f};
        std::vector<float> recv_buf(send_buf.size(), 0.0f);

        EXPECT_FALSE(backend_->allgather(
            send_buf.data(),
            recv_buf.data(),
            send_buf.size(),
            CollectiveDataType::FLOAT32));
    }

    TEST_F(Test__HostBackend, AllgatherCopiesDataForSingleDevice)
    {
        // For single-device groups, allgather copies send to recv
        auto group = createSingleCPUGroup();
        backend_->initialize(group);

        std::vector<float> send_buf = {1.0f, 2.0f, 3.0f, 4.0f};
        std::vector<float> recv_buf(send_buf.size(), 0.0f);

        backend_->allgather(
            send_buf.data(),
            recv_buf.data(),
            send_buf.size(),
            CollectiveDataType::FLOAT32);

        EXPECT_EQ(recv_buf, send_buf);
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // ReduceScatter Tests
    // ═══════════════════════════════════════════════════════════════════════════

    TEST_F(Test__HostBackend, ReduceScatterSucceedsWhenInitialized)
    {
        auto group = createSingleCPUGroup();
        backend_->initialize(group);

        std::vector<float> send_buf = {1.0f, 2.0f, 3.0f, 4.0f};
        std::vector<float> recv_buf(send_buf.size(), 0.0f);

        EXPECT_TRUE(backend_->reduceScatter(
            send_buf.data(),
            recv_buf.data(),
            send_buf.size(),
            CollectiveDataType::FLOAT32,
            CollectiveOp::ALLREDUCE_SUM));
    }

    TEST_F(Test__HostBackend, ReduceScatterFailsWhenNotInitialized)
    {
        std::vector<float> send_buf = {1.0f, 2.0f, 3.0f, 4.0f};
        std::vector<float> recv_buf(send_buf.size(), 0.0f);

        EXPECT_FALSE(backend_->reduceScatter(
            send_buf.data(),
            recv_buf.data(),
            send_buf.size(),
            CollectiveDataType::FLOAT32,
            CollectiveOp::ALLREDUCE_SUM));
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // Broadcast Tests
    // ═══════════════════════════════════════════════════════════════════════════

    TEST_F(Test__HostBackend, BroadcastSucceedsWhenInitialized)
    {
        auto group = createSingleCPUGroup();
        backend_->initialize(group);

        std::vector<float> buffer = {1.0f, 2.0f, 3.0f, 4.0f};
        EXPECT_TRUE(backend_->broadcast(
            buffer.data(),
            buffer.size(),
            CollectiveDataType::FLOAT32,
            0));
    }

    TEST_F(Test__HostBackend, BroadcastFailsWhenNotInitialized)
    {
        std::vector<float> buffer = {1.0f, 2.0f, 3.0f, 4.0f};
        EXPECT_FALSE(backend_->broadcast(
            buffer.data(),
            buffer.size(),
            CollectiveDataType::FLOAT32,
            0));
    }

    TEST_F(Test__HostBackend, BroadcastPreservesDataForSingleDevice)
    {
        // For single-device groups, broadcast is a no-op
        auto group = createSingleCPUGroup();
        backend_->initialize(group);

        std::vector<float> buffer = {1.0f, 2.0f, 3.0f, 4.0f};
        std::vector<float> expected = buffer;

        backend_->broadcast(
            buffer.data(),
            buffer.size(),
            CollectiveDataType::FLOAT32,
            0);

        EXPECT_EQ(buffer, expected);
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // Synchronize Tests
    // ═══════════════════════════════════════════════════════════════════════════

    TEST_F(Test__HostBackend, SynchronizeAlwaysSucceeds)
    {
        // Synchronize should succeed even without initialization
        // (CPU operations are inherently synchronous)
        EXPECT_TRUE(backend_->synchronize());

        // Also succeeds after initialization
        auto group = createSingleCPUGroup();
        backend_->initialize(group);
        EXPECT_TRUE(backend_->synchronize());

        // And after shutdown
        backend_->shutdown();
        EXPECT_TRUE(backend_->synchronize());
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // Multi-Device Tests
    // ═══════════════════════════════════════════════════════════════════════════

    TEST_F(Test__HostBackend, HandlesMultiCPUGroup)
    {
        // Test multi-device behavior with CPU devices (works without real GPUs)
        DeviceGroupBuilder builder;
        auto group = builder
            .setName("multi_cpu")
            .setScope(CollectiveScope::LOCAL)
            .addDevice(DeviceId::cpu())
            .addDevice(DeviceId::cpu())
            .setLocalRank(0)
            .build();

        EXPECT_TRUE(backend_->initialize(group));

        // Multi-device allreduce on CPU should work
        std::vector<float> buffer = {1.0f, 2.0f, 3.0f, 4.0f};
        EXPECT_TRUE(backend_->allreduce(
            buffer.data(),
            buffer.size(),
            CollectiveDataType::FLOAT32,
            CollectiveOp::ALLREDUCE_SUM));
    }

    TEST_F(Test__HostBackend, HandlesHeterogeneousCPUGroup)
    {
        // Test heterogeneous behavior with CPU devices (no real GPU needed)
        DeviceGroupBuilder builder;
        auto group = builder
            .setName("hetero_cpu")
            .setScope(CollectiveScope::LOCAL)
            .addDevice(DeviceId::cpu())
            .addDevice(DeviceId::cpu())
            .setLocalRank(0)
            .build();

        EXPECT_TRUE(backend_->initialize(group));

        // Heterogeneous operations should work
        std::vector<float> buffer = {1.0f, 2.0f, 3.0f, 4.0f};
        EXPECT_TRUE(backend_->allreduce(
            buffer.data(),
            buffer.size(),
            CollectiveDataType::FLOAT32,
            CollectiveOp::ALLREDUCE_SUM));
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // Data Type Tests
    // ═══════════════════════════════════════════════════════════════════════════

    TEST_F(Test__HostBackend, AllgatherWorksWithDifferentDataTypes)
    {
        auto group = createSingleCPUGroup();
        backend_->initialize(group);

        // FLOAT32
        {
            std::vector<float> send = {1.0f, 2.0f};
            std::vector<float> recv(2, 0.0f);
            EXPECT_TRUE(backend_->allgather(send.data(), recv.data(), 2, CollectiveDataType::FLOAT32));
            EXPECT_EQ(recv, send);
        }

        // INT32
        {
            std::vector<int32_t> send = {10, 20};
            std::vector<int32_t> recv(2, 0);
            EXPECT_TRUE(backend_->allgather(send.data(), recv.data(), 2, CollectiveDataType::INT32));
            EXPECT_EQ(recv, send);
        }

        // INT8
        {
            std::vector<int8_t> send = {1, 2, 3, 4};
            std::vector<int8_t> recv(4, 0);
            EXPECT_TRUE(backend_->allgather(send.data(), recv.data(), 4, CollectiveDataType::INT8));
            EXPECT_EQ(recv, send);
        }
    }

} // namespace llaminar2::test
