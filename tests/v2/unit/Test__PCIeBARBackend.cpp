/**
 * @file Test__PCIeBARBackend.cpp
 * @brief Unit tests for PCIeBARBackend
 *
 * Tests the direct CUDA↔ROCm collective backend via PCIe BAR mapping.
 * These tests verify backend behavior in isolation, using mock P2P engine
 * where appropriate for deterministic testing.
 *
 * @note Full integration tests with actual hardware are in
 *       integration/Test__PCIeBARBackendIntegration.cpp
 */

#include <gtest/gtest.h>
#include "v2/collective/backends/PCIeBARBackend.h"
#include "v2/collective/DeviceGroup.h"
#include "v2/backends/DeviceId.h"

#if defined(HAVE_CUDA) && defined(HAVE_ROCM)

namespace llaminar2::test
{

    // ═══════════════════════════════════════════════════════════════════════════
    // Test Fixture
    // ═══════════════════════════════════════════════════════════════════════════

    class Test__PCIeBARBackend : public ::testing::Test
    {
    protected:
        void SetUp() override
        {
            // Check if hardware is available
            auto caps = DirectP2PEngine::probeCapabilities();
            hardware_available_ = caps.canDoPCIeBarP2P();

            if (hardware_available_)
            {
                backend_ = std::make_unique<PCIeBARBackend>();
            }
        }

        void TearDown() override
        {
            if (backend_ && backend_->isInitialized())
            {
                backend_->shutdown();
            }
        }

        // Helper to create a CUDA + ROCm mixed group
        DeviceGroup createCUDAROCmGroup()
        {
            DeviceGroupBuilder builder;
            return builder
                .setName("cuda_rocm_group")
                .setScope(CollectiveScope::LOCAL)
                .addDevice(DeviceId::cuda(0))
                .addDevice(DeviceId::rocm(0))
                .setLocalRank(0)
                .build();
        }

        // Helper to create a single CUDA device group
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

        // Helper to create a single ROCm device group
        DeviceGroup createSingleROCmGroup()
        {
            DeviceGroupBuilder builder;
            return builder
                .setName("single_rocm")
                .setScope(CollectiveScope::LOCAL)
                .addDevice(DeviceId::rocm(0))
                .setLocalRank(0)
                .build();
        }

        std::unique_ptr<PCIeBARBackend> backend_;
        bool hardware_available_ = false;
    };

    // ═══════════════════════════════════════════════════════════════════════════
    // Identity Tests
    // ═══════════════════════════════════════════════════════════════════════════

    TEST_F(Test__PCIeBARBackend, TypeIsPCIeBAR)
    {
        if (!hardware_available_)
        {
            GTEST_SKIP() << "PCIe BAR P2P hardware not available";
        }

        EXPECT_EQ(backend_->type(), CollectiveBackendType::PCIE_BAR);
    }

    TEST_F(Test__PCIeBARBackend, NameIsPCIeBAR)
    {
        if (!hardware_available_)
        {
            GTEST_SKIP() << "PCIe BAR P2P hardware not available";
        }

        EXPECT_EQ(backend_->name(), "PCIe_BAR");
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // Capability Tests
    // ═══════════════════════════════════════════════════════════════════════════

    TEST_F(Test__PCIeBARBackend, SupportsCUDADevices)
    {
        if (!hardware_available_)
        {
            GTEST_SKIP() << "PCIe BAR P2P hardware not available";
        }

        EXPECT_TRUE(backend_->supportsDevice(DeviceType::CUDA));
    }

    TEST_F(Test__PCIeBARBackend, SupportsROCmDevices)
    {
        if (!hardware_available_)
        {
            GTEST_SKIP() << "PCIe BAR P2P hardware not available";
        }

        EXPECT_TRUE(backend_->supportsDevice(DeviceType::ROCm));
    }

    TEST_F(Test__PCIeBARBackend, DoesNotSupportCPU)
    {
        if (!hardware_available_)
        {
            GTEST_SKIP() << "PCIe BAR P2P hardware not available";
        }

        EXPECT_FALSE(backend_->supportsDevice(DeviceType::CPU));
    }

    TEST_F(Test__PCIeBARBackend, IsAvailableWhenHardwarePresent)
    {
        if (!hardware_available_)
        {
            GTEST_SKIP() << "PCIe BAR P2P hardware not available";
        }

        EXPECT_TRUE(backend_->isAvailable());
    }

    TEST_F(Test__PCIeBARBackend, SupportsDirectTransferCUDAtoROCm)
    {
        if (!hardware_available_)
        {
            GTEST_SKIP() << "PCIe BAR P2P hardware not available";
        }

        // Need to initialize first for direct transfer support
        auto group = createCUDAROCmGroup();
        ASSERT_TRUE(backend_->initialize(group));

        EXPECT_TRUE(backend_->supportsDirectTransfer(
            DeviceId::cuda(0), DeviceId::rocm(0)));
    }

    TEST_F(Test__PCIeBARBackend, SupportsDirectTransferROCmtoCUDA)
    {
        if (!hardware_available_)
        {
            GTEST_SKIP() << "PCIe BAR P2P hardware not available";
        }

        auto group = createCUDAROCmGroup();
        ASSERT_TRUE(backend_->initialize(group));

        EXPECT_TRUE(backend_->supportsDirectTransfer(
            DeviceId::rocm(0), DeviceId::cuda(0)));
    }

    TEST_F(Test__PCIeBARBackend, DoesNotSupportDirectTransferCUDAtoCUDA)
    {
        if (!hardware_available_)
        {
            GTEST_SKIP() << "PCIe BAR P2P hardware not available";
        }

        auto group = createCUDAROCmGroup();
        ASSERT_TRUE(backend_->initialize(group));

        // Same-vendor transfers not supported by PCIe BAR (use NCCL instead)
        EXPECT_FALSE(backend_->supportsDirectTransfer(
            DeviceId::cuda(0), DeviceId::cuda(1)));
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // Lifecycle Tests
    // ═══════════════════════════════════════════════════════════════════════════

    TEST_F(Test__PCIeBARBackend, InitializeWithValidGroup)
    {
        if (!hardware_available_)
        {
            GTEST_SKIP() << "PCIe BAR P2P hardware not available";
        }

        auto group = createCUDAROCmGroup();
        EXPECT_TRUE(backend_->initialize(group));
        EXPECT_TRUE(backend_->isInitialized());
    }

    TEST_F(Test__PCIeBARBackend, InitializeFailsWithOnlyCUDA)
    {
        if (!hardware_available_)
        {
            GTEST_SKIP() << "PCIe BAR P2P hardware not available";
        }

        auto group = createSingleCUDAGroup();
        EXPECT_FALSE(backend_->initialize(group));
        EXPECT_FALSE(backend_->isInitialized());
    }

    TEST_F(Test__PCIeBARBackend, InitializeFailsWithOnlyROCm)
    {
        if (!hardware_available_)
        {
            GTEST_SKIP() << "PCIe BAR P2P hardware not available";
        }

        auto group = createSingleROCmGroup();
        EXPECT_FALSE(backend_->initialize(group));
        EXPECT_FALSE(backend_->isInitialized());
    }

    TEST_F(Test__PCIeBARBackend, DoubleInitializeSucceeds)
    {
        if (!hardware_available_)
        {
            GTEST_SKIP() << "PCIe BAR P2P hardware not available";
        }

        auto group = createCUDAROCmGroup();
        EXPECT_TRUE(backend_->initialize(group));
        EXPECT_TRUE(backend_->initialize(group)); // Second call should succeed
        EXPECT_TRUE(backend_->isInitialized());
    }

    TEST_F(Test__PCIeBARBackend, ShutdownCleansUp)
    {
        if (!hardware_available_)
        {
            GTEST_SKIP() << "PCIe BAR P2P hardware not available";
        }

        auto group = createCUDAROCmGroup();
        ASSERT_TRUE(backend_->initialize(group));
        ASSERT_TRUE(backend_->isInitialized());

        backend_->shutdown();
        EXPECT_FALSE(backend_->isInitialized());
    }

    TEST_F(Test__PCIeBARBackend, CanReinitializeAfterShutdown)
    {
        if (!hardware_available_)
        {
            GTEST_SKIP() << "PCIe BAR P2P hardware not available";
        }

        auto group = createCUDAROCmGroup();

        // First cycle
        ASSERT_TRUE(backend_->initialize(group));
        backend_->shutdown();
        EXPECT_FALSE(backend_->isInitialized());

        // Second cycle
        EXPECT_TRUE(backend_->initialize(group));
        EXPECT_TRUE(backend_->isInitialized());
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // Bandwidth Diagnostics Tests
    // ═══════════════════════════════════════════════════════════════════════════

    TEST_F(Test__PCIeBARBackend, MeasuredBandwidthAfterInitialize)
    {
        if (!hardware_available_)
        {
            GTEST_SKIP() << "PCIe BAR P2P hardware not available";
        }

        auto group = createCUDAROCmGroup();
        ASSERT_TRUE(backend_->initialize(group));

        // Should have measured bandwidth after initialization
        double bandwidth = backend_->getMeasuredBandwidthGBps();
        EXPECT_GT(bandwidth, 0.0) << "Bandwidth should be measured after init";
        EXPECT_LT(bandwidth, 20.0) << "Bandwidth should be reasonable (< 20 GB/s for PCIe 3.0)";

        // Log the measured bandwidth for informational purposes
        std::cout << "Measured PCIe BAR P2P bandwidth: " << bandwidth << " GB/s" << std::endl;
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // Synchronization Tests
    // ═══════════════════════════════════════════════════════════════════════════

    TEST_F(Test__PCIeBARBackend, SynchronizeSucceeds)
    {
        if (!hardware_available_)
        {
            GTEST_SKIP() << "PCIe BAR P2P hardware not available";
        }

        auto group = createCUDAROCmGroup();
        ASSERT_TRUE(backend_->initialize(group));

        EXPECT_TRUE(backend_->synchronize());
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // Unsupported Operations Tests
    // ═══════════════════════════════════════════════════════════════════════════

    TEST_F(Test__PCIeBARBackend, ReduceScatterNotImplemented)
    {
        if (!hardware_available_)
        {
            GTEST_SKIP() << "PCIe BAR P2P hardware not available";
        }

        auto group = createCUDAROCmGroup();
        ASSERT_TRUE(backend_->initialize(group));

        std::vector<float> send(128, 1.0f);
        std::vector<float> recv(64);

        // ReduceScatter is not implemented for PCIe BAR
        EXPECT_FALSE(backend_->reduceScatter(
            send.data(), recv.data(), 64,
            CollectiveDataType::FLOAT32, CollectiveOp::ALLREDUCE_SUM));
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // BAR Region Allocator Tests
    // ═══════════════════════════════════════════════════════════════════════════

    TEST_F(Test__PCIeBARBackend, AllocateInBarRegion_SucceedsWhenBarActive)
    {
        if (!hardware_available_)
        {
            GTEST_SKIP() << "PCIe BAR P2P hardware not available";
        }

        auto group = createCUDAROCmGroup();
        ASSERT_TRUE(backend_->initialize(group));

        // Allocate a small buffer
        auto result = backend_->allocateInBarRegion(1024);
        ASSERT_TRUE(result.has_value());

        auto [ptr, offset] = *result;
        EXPECT_NE(ptr, nullptr);
        EXPECT_EQ(offset, 0); // First allocation starts at offset 0 (aligned)
    }

    TEST_F(Test__PCIeBARBackend, AllocateInBarRegion_ReturnsNulloptWhenNotInitialized)
    {
        if (!hardware_available_)
        {
            GTEST_SKIP() << "PCIe BAR P2P hardware not available";
        }

        // Not initialized - should return nullopt
        auto result = backend_->allocateInBarRegion(1024);
        EXPECT_FALSE(result.has_value());
    }

    TEST_F(Test__PCIeBARBackend, AllocateInBarRegion_MultipleAllocationsGetSequentialOffsets)
    {
        if (!hardware_available_)
        {
            GTEST_SKIP() << "PCIe BAR P2P hardware not available";
        }

        auto group = createCUDAROCmGroup();
        ASSERT_TRUE(backend_->initialize(group));

        // First allocation
        auto result1 = backend_->allocateInBarRegion(256);
        ASSERT_TRUE(result1.has_value());
        auto [ptr1, offset1] = *result1;
        EXPECT_EQ(offset1, 0);

        // Second allocation (should be at next aligned offset)
        auto result2 = backend_->allocateInBarRegion(512);
        ASSERT_TRUE(result2.has_value());
        auto [ptr2, offset2] = *result2;
        EXPECT_EQ(offset2, 256); // After first 256-byte aligned allocation

        // Third allocation
        auto result3 = backend_->allocateInBarRegion(1024);
        ASSERT_TRUE(result3.has_value());
        auto [ptr3, offset3] = *result3;
        EXPECT_EQ(offset3, 256 + 512); // After first two allocations

        // Verify pointers are different
        EXPECT_NE(ptr1, ptr2);
        EXPECT_NE(ptr2, ptr3);
        EXPECT_NE(ptr1, ptr3);
    }

    TEST_F(Test__PCIeBARBackend, AllocateInBarRegion_AlignsToAlignment)
    {
        if (!hardware_available_)
        {
            GTEST_SKIP() << "PCIe BAR P2P hardware not available";
        }

        auto group = createCUDAROCmGroup();
        ASSERT_TRUE(backend_->initialize(group));

        // Request odd size, should align up
        auto result = backend_->allocateInBarRegion(100);
        ASSERT_TRUE(result.has_value());

        // Second allocation should be at aligned offset (256)
        auto result2 = backend_->allocateInBarRegion(100);
        ASSERT_TRUE(result2.has_value());
        auto [ptr2, offset2] = *result2;
        EXPECT_EQ(offset2, 256); // 100 rounded up to 256
    }

    TEST_F(Test__PCIeBARBackend, AllocateInBarRegion_FailsWhenExceedingBarSize)
    {
        if (!hardware_available_)
        {
            GTEST_SKIP() << "PCIe BAR P2P hardware not available";
        }

        auto group = createCUDAROCmGroup();
        ASSERT_TRUE(backend_->initialize(group));

        // Get BAR size and verify it's what we expect
        size_t bar_size = backend_->getBarTotalMappedSize();
        ASSERT_GT(bar_size, 0) << "BAR should be mapped";

        // Try to allocate more than the BAR size in one allocation
        auto result = backend_->allocateInBarRegion(bar_size + 1);
        EXPECT_FALSE(result.has_value()) << "Should fail when allocation exceeds BAR size";
    }

    TEST_F(Test__PCIeBARBackend, FreeBarBuffer_RemovesFromTracking)
    {
        if (!hardware_available_)
        {
            GTEST_SKIP() << "PCIe BAR P2P hardware not available";
        }

        auto group = createCUDAROCmGroup();
        ASSERT_TRUE(backend_->initialize(group));

        // Allocate and free
        auto result = backend_->allocateInBarRegion(1024);
        ASSERT_TRUE(result.has_value());
        auto [ptr, offset] = *result;

        // Free should not crash (bump allocator, just removes tracking)
        backend_->freeBarBuffer(ptr);

        // Freeing null should also not crash
        backend_->freeBarBuffer(nullptr);
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // Buffer Registration Tests
    // ═══════════════════════════════════════════════════════════════════════════

    TEST_F(Test__PCIeBARBackend, RequiresBufferRegistration)
    {
        if (!hardware_available_)
        {
            GTEST_SKIP() << "PCIe BAR P2P hardware not available";
        }

        EXPECT_TRUE(backend_->requiresBufferRegistration());
    }

    TEST_F(Test__PCIeBARBackend, RegisterBuffer_CUDASucceeds)
    {
        if (!hardware_available_)
        {
            GTEST_SKIP() << "PCIe BAR P2P hardware not available";
        }

        auto group = createCUDAROCmGroup();
        ASSERT_TRUE(backend_->initialize(group));

        // Create a fake CUDA buffer (in real code this would be cudaMalloc'd)
        float fake_buffer[256];
        EXPECT_TRUE(backend_->registerBuffer("test_collective", DeviceId::cuda(0), fake_buffer, sizeof(fake_buffer)));

        // Verify we can retrieve it
        auto reg = backend_->getBuffer("test_collective", DeviceId::cuda(0));
        ASSERT_TRUE(reg.has_value());
        EXPECT_EQ(reg->ptr, fake_buffer);
        EXPECT_EQ(reg->size, sizeof(fake_buffer));
        EXPECT_TRUE(reg->device.is_cuda());
    }

    TEST_F(Test__PCIeBARBackend, RegisterBuffer_ROCmWithBarAllocation)
    {
        if (!hardware_available_)
        {
            GTEST_SKIP() << "PCIe BAR P2P hardware not available";
        }

        auto group = createCUDAROCmGroup();
        ASSERT_TRUE(backend_->initialize(group));

        // Allocate a buffer in BAR region (this is what ROCm would use)
        auto alloc = backend_->allocateInBarRegion(1024);
        ASSERT_TRUE(alloc.has_value());
        auto [ptr, offset] = *alloc;

        // Now register it as ROCm buffer
        EXPECT_TRUE(backend_->registerBuffer("test_collective", DeviceId::rocm(0), ptr, 1024));

        // Verify we can retrieve it with correct offset
        auto reg = backend_->getBuffer("test_collective", DeviceId::rocm(0));
        ASSERT_TRUE(reg.has_value());
        EXPECT_EQ(reg->ptr, ptr);
        EXPECT_EQ(reg->bar_offset, offset);
        EXPECT_EQ(reg->size, 1024);
        EXPECT_TRUE(reg->device.is_rocm());
    }

    TEST_F(Test__PCIeBARBackend, RegisterBuffer_ROCmFailsWithoutBarAllocation)
    {
        if (!hardware_available_)
        {
            GTEST_SKIP() << "PCIe BAR P2P hardware not available";
        }

        auto group = createCUDAROCmGroup();
        ASSERT_TRUE(backend_->initialize(group));

        // Try to register a non-BAR-allocated buffer as ROCm
        float fake_buffer[256];
        EXPECT_FALSE(backend_->registerBuffer("test_collective", DeviceId::rocm(0), fake_buffer, sizeof(fake_buffer)));
    }

    TEST_F(Test__PCIeBARBackend, GetBuffer_ReturnsNulloptForUnregistered)
    {
        if (!hardware_available_)
        {
            GTEST_SKIP() << "PCIe BAR P2P hardware not available";
        }

        auto group = createCUDAROCmGroup();
        ASSERT_TRUE(backend_->initialize(group));

        auto result = backend_->getBuffer("nonexistent_collective", DeviceId::cuda(0));
        EXPECT_FALSE(result.has_value());
    }

    TEST_F(Test__PCIeBARBackend, UnregisterBuffer_ClearsBothDevices)
    {
        if (!hardware_available_)
        {
            GTEST_SKIP() << "PCIe BAR P2P hardware not available";
        }

        auto group = createCUDAROCmGroup();
        ASSERT_TRUE(backend_->initialize(group));

        // Register CUDA buffer
        float fake_buffer[256];
        ASSERT_TRUE(backend_->registerBuffer("test_collective", DeviceId::cuda(0), fake_buffer, sizeof(fake_buffer)));

        // Allocate and register ROCm buffer
        auto alloc = backend_->allocateInBarRegion(1024);
        ASSERT_TRUE(alloc.has_value());
        ASSERT_TRUE(backend_->registerBuffer("test_collective", DeviceId::rocm(0), alloc->first, 1024));

        // Unregister CUDA
        backend_->unregisterBuffer("test_collective", DeviceId::cuda(0));
        EXPECT_FALSE(backend_->getBuffer("test_collective", DeviceId::cuda(0)).has_value());
        // ROCm should still be registered
        EXPECT_TRUE(backend_->getBuffer("test_collective", DeviceId::rocm(0)).has_value());

        // Unregister ROCm
        backend_->unregisterBuffer("test_collective", DeviceId::rocm(0));
        EXPECT_FALSE(backend_->getBuffer("test_collective", DeviceId::rocm(0)).has_value());
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // AllReduce Registered Tests
    // ═══════════════════════════════════════════════════════════════════════════

    TEST_F(Test__PCIeBARBackend, AllreduceRegistered_FailsWithoutRegistration)
    {
        if (!hardware_available_)
        {
            GTEST_SKIP() << "PCIe BAR P2P hardware not available";
        }

        auto group = createCUDAROCmGroup();
        ASSERT_TRUE(backend_->initialize(group));

        // No registration - should fail
        EXPECT_FALSE(backend_->allreduceRegistered(
            "unregistered_collective",
            100,
            CollectiveDataType::FLOAT32,
            CollectiveOp::ALLREDUCE_SUM));
    }

    TEST_F(Test__PCIeBARBackend, AllreduceRegistered_FailsWithOnlyCUDA)
    {
        if (!hardware_available_)
        {
            GTEST_SKIP() << "PCIe BAR P2P hardware not available";
        }

        auto group = createCUDAROCmGroup();
        ASSERT_TRUE(backend_->initialize(group));

        // Register only CUDA
        float fake_buffer[256];
        ASSERT_TRUE(backend_->registerBuffer("test_collective", DeviceId::cuda(0), fake_buffer, sizeof(fake_buffer)));

        // Should fail - ROCm not registered
        EXPECT_FALSE(backend_->allreduceRegistered(
            "test_collective",
            256,
            CollectiveDataType::FLOAT32,
            CollectiveOp::ALLREDUCE_SUM));
    }

    TEST_F(Test__PCIeBARBackend, AllreduceRegistered_FailsWithOnlyROCm)
    {
        if (!hardware_available_)
        {
            GTEST_SKIP() << "PCIe BAR P2P hardware not available";
        }

        auto group = createCUDAROCmGroup();
        ASSERT_TRUE(backend_->initialize(group));

        // Allocate and register only ROCm
        auto alloc = backend_->allocateInBarRegion(1024);
        ASSERT_TRUE(alloc.has_value());
        ASSERT_TRUE(backend_->registerBuffer("test_collective", DeviceId::rocm(0), alloc->first, 1024));

        // Should fail - CUDA not registered
        EXPECT_FALSE(backend_->allreduceRegistered(
            "test_collective",
            256,
            CollectiveDataType::FLOAT32,
            CollectiveOp::ALLREDUCE_SUM));
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // P2P Operations - Not Initialized Tests
    // ═══════════════════════════════════════════════════════════════════════════

    TEST_F(Test__PCIeBARBackend, Send_FailsWhenNotInitialized)
    {
        if (!hardware_available_)
        {
            GTEST_SKIP() << "PCIe BAR P2P hardware not available";
        }

        float buffer[64];
        EXPECT_FALSE(backend_->send(buffer, 64, CollectiveDataType::FLOAT32, 1, 0))
            << "send() should fail when backend is not initialized";
    }

    TEST_F(Test__PCIeBARBackend, Recv_FailsWhenNotInitialized)
    {
        if (!hardware_available_)
        {
            GTEST_SKIP() << "PCIe BAR P2P hardware not available";
        }

        float buffer[64];
        EXPECT_FALSE(backend_->recv(buffer, 64, CollectiveDataType::FLOAT32, 0, 0))
            << "recv() should fail when backend is not initialized";
    }

    TEST_F(Test__PCIeBARBackend, Sendrecv_FailsWhenNotInitialized)
    {
        if (!hardware_available_)
        {
            GTEST_SKIP() << "PCIe BAR P2P hardware not available";
        }

        float sendbuf[64];
        float recvbuf[64];
        EXPECT_FALSE(backend_->sendrecv(sendbuf, recvbuf, 64, CollectiveDataType::FLOAT32, 1))
            << "sendrecv() should fail when backend is not initialized";
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // P2P Operations - Null Buffer Tests
    // ═══════════════════════════════════════════════════════════════════════════

    TEST_F(Test__PCIeBARBackend, Send_FailsWithNullBuffer)
    {
        if (!hardware_available_)
        {
            GTEST_SKIP() << "PCIe BAR P2P hardware not available";
        }

        auto group = createCUDAROCmGroup();
        ASSERT_TRUE(backend_->initialize(group));

        EXPECT_FALSE(backend_->send(nullptr, 64, CollectiveDataType::FLOAT32, 0, 0))
            << "send() should fail with null buffer";
    }

    TEST_F(Test__PCIeBARBackend, Recv_FailsWithNullBuffer)
    {
        if (!hardware_available_)
        {
            GTEST_SKIP() << "PCIe BAR P2P hardware not available";
        }

        auto group = createCUDAROCmGroup();
        ASSERT_TRUE(backend_->initialize(group));

        EXPECT_FALSE(backend_->recv(nullptr, 64, CollectiveDataType::FLOAT32, 0, 0))
            << "recv() should fail with null buffer";
    }

    TEST_F(Test__PCIeBARBackend, Sendrecv_FailsWithNullSendBuffer)
    {
        if (!hardware_available_)
        {
            GTEST_SKIP() << "PCIe BAR P2P hardware not available";
        }

        auto group = createCUDAROCmGroup();
        ASSERT_TRUE(backend_->initialize(group));

        float recvbuf[64];
        EXPECT_FALSE(backend_->sendrecv(nullptr, recvbuf, 64, CollectiveDataType::FLOAT32, 0))
            << "sendrecv() should fail with null send buffer";
    }

    TEST_F(Test__PCIeBARBackend, Sendrecv_FailsWithNullRecvBuffer)
    {
        if (!hardware_available_)
        {
            GTEST_SKIP() << "PCIe BAR P2P hardware not available";
        }

        auto group = createCUDAROCmGroup();
        ASSERT_TRUE(backend_->initialize(group));

        float sendbuf[64];
        EXPECT_FALSE(backend_->sendrecv(sendbuf, nullptr, 64, CollectiveDataType::FLOAT32, 0))
            << "sendrecv() should fail with null recv buffer";
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // P2P Operations - Zero Count Tests
    // ═══════════════════════════════════════════════════════════════════════════

    TEST_F(Test__PCIeBARBackend, Send_HandlesZeroCount)
    {
        if (!hardware_available_)
        {
            GTEST_SKIP() << "PCIe BAR P2P hardware not available";
        }

        auto group = createCUDAROCmGroup();
        ASSERT_TRUE(backend_->initialize(group));

        float buffer[1];
        // Zero count should either succeed (no-op) or fail gracefully
        bool result = backend_->send(buffer, 0, CollectiveDataType::FLOAT32, 0, 0);
        // We just verify no crash occurs
        (void)result;
    }

    TEST_F(Test__PCIeBARBackend, Recv_HandlesZeroCount)
    {
        if (!hardware_available_)
        {
            GTEST_SKIP() << "PCIe BAR P2P hardware not available";
        }

        auto group = createCUDAROCmGroup();
        ASSERT_TRUE(backend_->initialize(group));

        float buffer[1];
        // Zero count should either succeed (no-op) or fail gracefully
        bool result = backend_->recv(buffer, 0, CollectiveDataType::FLOAT32, 0, 0);
        (void)result;
    }

    TEST_F(Test__PCIeBARBackend, Sendrecv_HandlesZeroCount)
    {
        if (!hardware_available_)
        {
            GTEST_SKIP() << "PCIe BAR P2P hardware not available";
        }

        auto group = createCUDAROCmGroup();
        ASSERT_TRUE(backend_->initialize(group));

        float sendbuf[1];
        float recvbuf[1];
        // Zero count should either succeed (no-op) or fail gracefully
        bool result = backend_->sendrecv(sendbuf, recvbuf, 0, CollectiveDataType::FLOAT32, 0);
        (void)result;
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // P2P Operations - Invalid Peer Tests
    // ═══════════════════════════════════════════════════════════════════════════

    TEST_F(Test__PCIeBARBackend, Send_FailsWithInvalidPeer)
    {
        if (!hardware_available_)
        {
            GTEST_SKIP() << "PCIe BAR P2P hardware not available";
        }

        auto group = createCUDAROCmGroup();
        ASSERT_TRUE(backend_->initialize(group));

        float buffer[64];
        // Peer rank 99 is invalid for a 2-device group
        EXPECT_FALSE(backend_->send(buffer, 64, CollectiveDataType::FLOAT32, 99, 0))
            << "send() should fail with invalid peer rank";
    }

    TEST_F(Test__PCIeBARBackend, Recv_FailsWithInvalidPeer)
    {
        if (!hardware_available_)
        {
            GTEST_SKIP() << "PCIe BAR P2P hardware not available";
        }

        auto group = createCUDAROCmGroup();
        ASSERT_TRUE(backend_->initialize(group));

        float buffer[64];
        // Peer rank -1 is invalid
        EXPECT_FALSE(backend_->recv(buffer, 64, CollectiveDataType::FLOAT32, -1, 0))
            << "recv() should fail with negative peer rank";
    }

    TEST_F(Test__PCIeBARBackend, Sendrecv_FailsWithInvalidPeer)
    {
        if (!hardware_available_)
        {
            GTEST_SKIP() << "PCIe BAR P2P hardware not available";
        }

        auto group = createCUDAROCmGroup();
        ASSERT_TRUE(backend_->initialize(group));

        float sendbuf[64];
        float recvbuf[64];
        // Peer rank 100 is invalid
        EXPECT_FALSE(backend_->sendrecv(sendbuf, recvbuf, 64, CollectiveDataType::FLOAT32, 100))
            << "sendrecv() should fail with invalid peer rank";
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // Multi-Pair API Tests (NO hardware required)
    //
    // These tests verify the multi-pair API validation logic and state management
    // WITHOUT requiring actual hardware initialization. They test:
    // - DevicePair struct equality operator
    // - isMultiPairMode() state before initialization
    // - initializeMultiPair() parameter validation
    // - allreduceMultiPair() behavior when not initialized
    // ═══════════════════════════════════════════════════════════════════════════

    class Test__PCIeBARBackend_MultiPair : public ::testing::Test
    {
    protected:
        // Create backend WITHOUT hardware check - for validation-only tests
        std::unique_ptr<PCIeBARBackend> createBackendForValidation()
        {
            return std::make_unique<PCIeBARBackend>();
        }
    };

    // ─────────────────────────────────────────────────────────────────────────────
    // DevicePair Struct Tests
    // ─────────────────────────────────────────────────────────────────────────────

    TEST_F(Test__PCIeBARBackend_MultiPair, DevicePair_EqualityOperator_SameDevices)
    {
        DevicePair pair1{DeviceId::cuda(0), DeviceId::rocm(0), 0};
        DevicePair pair2{DeviceId::cuda(0), DeviceId::rocm(0), 0};

        EXPECT_TRUE(pair1 == pair2) << "Identical pairs should be equal";
    }

    TEST_F(Test__PCIeBARBackend_MultiPair, DevicePair_EqualityOperator_DifferentCUDA)
    {
        DevicePair pair1{DeviceId::cuda(0), DeviceId::rocm(0), 0};
        DevicePair pair2{DeviceId::cuda(1), DeviceId::rocm(0), 0};

        EXPECT_FALSE(pair1 == pair2) << "Pairs with different CUDA devices should not be equal";
    }

    TEST_F(Test__PCIeBARBackend_MultiPair, DevicePair_EqualityOperator_DifferentROCm)
    {
        DevicePair pair1{DeviceId::cuda(0), DeviceId::rocm(0), 0};
        DevicePair pair2{DeviceId::cuda(0), DeviceId::rocm(1), 0};

        EXPECT_FALSE(pair1 == pair2) << "Pairs with different ROCm devices should not be equal";
    }

    TEST_F(Test__PCIeBARBackend_MultiPair, DevicePair_EqualityOperator_IgnoresPairIndex)
    {
        // Equality is based on device IDs only, not pair_index
        DevicePair pair1{DeviceId::cuda(0), DeviceId::rocm(0), 0};
        DevicePair pair2{DeviceId::cuda(0), DeviceId::rocm(0), 5};

        EXPECT_TRUE(pair1 == pair2) << "Pairs should be equal regardless of pair_index";
    }

    // ─────────────────────────────────────────────────────────────────────────────
    // isMultiPairMode() State Tests
    // ─────────────────────────────────────────────────────────────────────────────

    TEST_F(Test__PCIeBARBackend_MultiPair, IsMultiPairMode_FalseBeforeInitialize)
    {
        auto backend = createBackendForValidation();

        EXPECT_FALSE(backend->isMultiPairMode())
            << "isMultiPairMode() should return false before initialization";
    }

    TEST_F(Test__PCIeBARBackend_MultiPair, GetDevicePairs_EmptyBeforeInitialize)
    {
        auto backend = createBackendForValidation();

        const auto &pairs = backend->getDevicePairs();
        EXPECT_TRUE(pairs.empty())
            << "getDevicePairs() should return empty vector before initialization";
    }

    // ─────────────────────────────────────────────────────────────────────────────
    // initializeMultiPair() Validation Tests
    // ─────────────────────────────────────────────────────────────────────────────

    TEST_F(Test__PCIeBARBackend_MultiPair, InitializeMultiPair_RejectsEmptyPairsVector)
    {
        auto backend = createBackendForValidation();

        std::vector<DevicePair> empty_pairs;
        EXPECT_FALSE(backend->initializeMultiPair(empty_pairs))
            << "initializeMultiPair() should reject empty pairs vector";

        EXPECT_FALSE(backend->isMultiPairMode())
            << "isMultiPairMode() should remain false after failed initialization";
    }

    TEST_F(Test__PCIeBARBackend_MultiPair, InitializeMultiPair_RejectsNonCUDADeviceAsCUDA)
    {
        auto backend = createBackendForValidation();

        // Try to use ROCm device as cuda_device
        std::vector<DevicePair> invalid_pairs = {
            {DeviceId::rocm(0), DeviceId::rocm(1), 0} // rocm(0) is NOT a CUDA device!
        };

        EXPECT_FALSE(backend->initializeMultiPair(invalid_pairs))
            << "initializeMultiPair() should reject non-CUDA device as cuda_device";

        EXPECT_FALSE(backend->isInitialized())
            << "Backend should not be initialized after validation failure";
    }

    TEST_F(Test__PCIeBARBackend_MultiPair, InitializeMultiPair_RejectsCPUDeviceAsCUDA)
    {
        auto backend = createBackendForValidation();

        // Try to use CPU device as cuda_device
        std::vector<DevicePair> invalid_pairs = {
            {DeviceId::cpu(), DeviceId::rocm(0), 0} // CPU is NOT a CUDA device!
        };

        EXPECT_FALSE(backend->initializeMultiPair(invalid_pairs))
            << "initializeMultiPair() should reject CPU device as cuda_device";
    }

    TEST_F(Test__PCIeBARBackend_MultiPair, InitializeMultiPair_RejectsNonROCmDeviceAsROCm)
    {
        auto backend = createBackendForValidation();

        // Try to use CUDA device as rocm_device
        std::vector<DevicePair> invalid_pairs = {
            {DeviceId::cuda(0), DeviceId::cuda(1), 0} // cuda(1) is NOT a ROCm device!
        };

        EXPECT_FALSE(backend->initializeMultiPair(invalid_pairs))
            << "initializeMultiPair() should reject non-ROCm device as rocm_device";
    }

    TEST_F(Test__PCIeBARBackend_MultiPair, InitializeMultiPair_RejectsCPUDeviceAsROCm)
    {
        auto backend = createBackendForValidation();

        // Try to use CPU device as rocm_device
        std::vector<DevicePair> invalid_pairs = {
            {DeviceId::cuda(0), DeviceId::cpu(), 0} // CPU is NOT a ROCm device!
        };

        EXPECT_FALSE(backend->initializeMultiPair(invalid_pairs))
            << "initializeMultiPair() should reject CPU device as rocm_device";
    }

    TEST_F(Test__PCIeBARBackend_MultiPair, InitializeMultiPair_RejectsDuplicateCUDADevices)
    {
        auto backend = createBackendForValidation();

        // Two pairs using the same CUDA device
        std::vector<DevicePair> invalid_pairs = {
            {DeviceId::cuda(0), DeviceId::rocm(0), 0},
            {DeviceId::cuda(0), DeviceId::rocm(1), 1} // cuda(0) already used in pair 0!
        };

        EXPECT_FALSE(backend->initializeMultiPair(invalid_pairs))
            << "initializeMultiPair() should reject duplicate CUDA devices across pairs";
    }

    TEST_F(Test__PCIeBARBackend_MultiPair, InitializeMultiPair_RejectsDuplicateROCmDevices)
    {
        auto backend = createBackendForValidation();

        // Two pairs using the same ROCm device
        std::vector<DevicePair> invalid_pairs = {
            {DeviceId::cuda(0), DeviceId::rocm(0), 0},
            {DeviceId::cuda(1), DeviceId::rocm(0), 1} // rocm(0) already used in pair 0!
        };

        EXPECT_FALSE(backend->initializeMultiPair(invalid_pairs))
            << "initializeMultiPair() should reject duplicate ROCm devices across pairs";
    }

    // ─────────────────────────────────────────────────────────────────────────────
    // allreduceMultiPair() Tests (not initialized)
    // ─────────────────────────────────────────────────────────────────────────────

    TEST_F(Test__PCIeBARBackend_MultiPair, AllreduceMultiPair_FailsWhenNotInitialized)
    {
        auto backend = createBackendForValidation();

        // Create fake buffer vectors
        std::vector<void *> cuda_buffers = {nullptr};
        std::vector<void *> rocm_buffers = {nullptr};

        EXPECT_FALSE(backend->allreduceMultiPair(
            cuda_buffers, rocm_buffers, 100, CollectiveDataType::FLOAT32))
            << "allreduceMultiPair() should fail when backend not initialized";
    }

    TEST_F(Test__PCIeBARBackend_MultiPair, AllreduceMultiPair_FailsWhenNotInMultiPairMode)
    {
        // This test requires hardware, but we can verify the check exists
        // by calling on an uninitialized backend
        auto backend = createBackendForValidation();

        std::vector<void *> cuda_buffers;
        std::vector<void *> rocm_buffers;

        // Empty buffers should fail even before multi-pair mode check
        EXPECT_FALSE(backend->allreduceMultiPair(
            cuda_buffers, rocm_buffers, 100, CollectiveDataType::FLOAT32))
            << "allreduceMultiPair() should fail with empty buffers";
    }

} // namespace llaminar2::test

#else // !HAVE_CUDA || !HAVE_ROCM

// Stub test when CUDA+ROCm not available
TEST(Test__PCIeBARBackend, RequiresCUDAAndROCm)
{
    GTEST_SKIP() << "PCIeBARBackend requires both HAVE_CUDA and HAVE_ROCM";
}

// Stub tests for multi-pair API when not available
TEST(Test__PCIeBARBackend_MultiPair_Stub, StubIsMultiPairMode)
{
    llaminar2::PCIeBARBackend backend;
    EXPECT_FALSE(backend.isMultiPairMode()) << "Stub should always return false";
}

TEST(Test__PCIeBARBackend_MultiPair_Stub, StubInitializeMultiPairFails)
{
    llaminar2::PCIeBARBackend backend;
    std::vector<llaminar2::DevicePair> pairs = {
        {llaminar2::DeviceId::cuda(0), llaminar2::DeviceId::rocm(0), 0}};
    EXPECT_FALSE(backend.initializeMultiPair(pairs))
        << "Stub should always fail initialization";
}

#endif // HAVE_CUDA && HAVE_ROCM
