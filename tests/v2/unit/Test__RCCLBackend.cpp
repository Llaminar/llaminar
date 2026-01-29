/**
 * @file Test__RCCLBackend.cpp
 * @brief Unit tests for RCCLBackend P2P operations
 *
 * Tests the point-to-point send/recv/sendrecv primitives for RCCL backend.
 * Validates error handling for uninitialized state, invalid parameters,
 * and edge cases.
 *
 * @note Full integration tests with actual RCCL operations are in
 *       integration/Test__CrossBackendP2P.cpp
 *
 * @author David Sanftenberg
 * @date January 2026
 */

#include <gtest/gtest.h>
#include "v2/collective/backends/RCCLBackend.h"
#include "v2/collective/DeviceGroup.h"
#include "v2/backends/DeviceId.h"

#ifdef HAVE_ROCM

namespace llaminar2::test
{

    // ═══════════════════════════════════════════════════════════════════════════
    // Test Fixture
    // ═══════════════════════════════════════════════════════════════════════════

    class Test__RCCLBackend : public ::testing::Test
    {
    protected:
        void SetUp() override
        {
            backend_ = std::make_unique<RCCLBackend>();
            hardware_available_ = backend_->isAvailable();
        }

        void TearDown() override
        {
            if (backend_ && backend_->isInitialized())
            {
                backend_->shutdown();
            }
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

        // Helper to create a 2-device ROCm group (for P2P tests)
        DeviceGroup createTwoROCmGroup()
        {
            DeviceGroupBuilder builder;
            return builder
                .setName("two_rocm")
                .setScope(CollectiveScope::LOCAL)
                .addDevice(DeviceId::rocm(0))
                .addDevice(DeviceId::rocm(1))
                .setLocalRank(0)
                .build();
        }

        std::unique_ptr<RCCLBackend> backend_;
        bool hardware_available_ = false;
    };

    // ═══════════════════════════════════════════════════════════════════════════
    // Identity Tests
    // ═══════════════════════════════════════════════════════════════════════════

    TEST_F(Test__RCCLBackend, TypeIsRCCL)
    {
        EXPECT_EQ(backend_->type(), CollectiveBackendType::RCCL);
    }

    TEST_F(Test__RCCLBackend, NameIsRCCL)
    {
        EXPECT_EQ(backend_->name(), "RCCL");
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // Capability Tests
    // ═══════════════════════════════════════════════════════════════════════════

    TEST_F(Test__RCCLBackend, SupportsROCmDevices)
    {
        EXPECT_TRUE(backend_->supportsDevice(DeviceType::ROCm));
    }

    TEST_F(Test__RCCLBackend, DoesNotSupportCUDA)
    {
        EXPECT_FALSE(backend_->supportsDevice(DeviceType::CUDA));
    }

    TEST_F(Test__RCCLBackend, DoesNotSupportCPU)
    {
        EXPECT_FALSE(backend_->supportsDevice(DeviceType::CPU));
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // P2P Operations - Not Initialized Tests
    // ═══════════════════════════════════════════════════════════════════════════

    TEST_F(Test__RCCLBackend, Send_FailsWhenNotInitialized)
    {
        if (!hardware_available_)
        {
            GTEST_SKIP() << "RCCL hardware not available";
        }

        float buffer[64];
        EXPECT_FALSE(backend_->send(buffer, 64, CollectiveDataType::FLOAT32, 1, 0))
            << "send() should fail when backend is not initialized";
    }

    TEST_F(Test__RCCLBackend, Recv_FailsWhenNotInitialized)
    {
        if (!hardware_available_)
        {
            GTEST_SKIP() << "RCCL hardware not available";
        }

        float buffer[64];
        EXPECT_FALSE(backend_->recv(buffer, 64, CollectiveDataType::FLOAT32, 0, 0))
            << "recv() should fail when backend is not initialized";
    }

    TEST_F(Test__RCCLBackend, Sendrecv_FailsWhenNotInitialized)
    {
        if (!hardware_available_)
        {
            GTEST_SKIP() << "RCCL hardware not available";
        }

        float sendbuf[64];
        float recvbuf[64];
        EXPECT_FALSE(backend_->sendrecv(sendbuf, recvbuf, 64, CollectiveDataType::FLOAT32, 1))
            << "sendrecv() should fail when backend is not initialized";
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // P2P Operations - Null Buffer Tests
    // ═══════════════════════════════════════════════════════════════════════════

    TEST_F(Test__RCCLBackend, Send_FailsWithNullBuffer)
    {
        if (!hardware_available_)
        {
            GTEST_SKIP() << "RCCL hardware not available";
        }

        auto group = createSingleROCmGroup();
        if (!backend_->initialize(group))
        {
            GTEST_SKIP() << "Backend initialization failed";
        }

        EXPECT_FALSE(backend_->send(nullptr, 64, CollectiveDataType::FLOAT32, 0, 0))
            << "send() should fail with null buffer";
    }

    TEST_F(Test__RCCLBackend, Recv_FailsWithNullBuffer)
    {
        if (!hardware_available_)
        {
            GTEST_SKIP() << "RCCL hardware not available";
        }

        auto group = createSingleROCmGroup();
        if (!backend_->initialize(group))
        {
            GTEST_SKIP() << "Backend initialization failed";
        }

        EXPECT_FALSE(backend_->recv(nullptr, 64, CollectiveDataType::FLOAT32, 0, 0))
            << "recv() should fail with null buffer";
    }

    TEST_F(Test__RCCLBackend, Sendrecv_FailsWithNullSendBuffer)
    {
        if (!hardware_available_)
        {
            GTEST_SKIP() << "RCCL hardware not available";
        }

        auto group = createSingleROCmGroup();
        if (!backend_->initialize(group))
        {
            GTEST_SKIP() << "Backend initialization failed";
        }

        float recvbuf[64];
        EXPECT_FALSE(backend_->sendrecv(nullptr, recvbuf, 64, CollectiveDataType::FLOAT32, 0))
            << "sendrecv() should fail with null send buffer";
    }

    TEST_F(Test__RCCLBackend, Sendrecv_FailsWithNullRecvBuffer)
    {
        if (!hardware_available_)
        {
            GTEST_SKIP() << "RCCL hardware not available";
        }

        auto group = createSingleROCmGroup();
        if (!backend_->initialize(group))
        {
            GTEST_SKIP() << "Backend initialization failed";
        }

        float sendbuf[64];
        EXPECT_FALSE(backend_->sendrecv(sendbuf, nullptr, 64, CollectiveDataType::FLOAT32, 0))
            << "sendrecv() should fail with null recv buffer";
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // P2P Operations - Zero Count Tests
    // ═══════════════════════════════════════════════════════════════════════════

    TEST_F(Test__RCCLBackend, Send_HandlesZeroCount)
    {
        if (!hardware_available_)
        {
            GTEST_SKIP() << "RCCL hardware not available";
        }

        auto group = createSingleROCmGroup();
        if (!backend_->initialize(group))
        {
            GTEST_SKIP() << "Backend initialization failed";
        }

        float buffer[1];
        // Zero count should either succeed (no-op) or fail gracefully
        // The exact behavior depends on RCCL implementation
        bool result = backend_->send(buffer, 0, CollectiveDataType::FLOAT32, 0, 0);
        // We just verify no crash occurs - result can be true (no-op success) or false (rejected)
        (void)result;
    }

    TEST_F(Test__RCCLBackend, Recv_HandlesZeroCount)
    {
        if (!hardware_available_)
        {
            GTEST_SKIP() << "RCCL hardware not available";
        }

        auto group = createSingleROCmGroup();
        if (!backend_->initialize(group))
        {
            GTEST_SKIP() << "Backend initialization failed";
        }

        float buffer[1];
        // Zero count should either succeed (no-op) or fail gracefully
        bool result = backend_->recv(buffer, 0, CollectiveDataType::FLOAT32, 0, 0);
        (void)result;
    }

    TEST_F(Test__RCCLBackend, Sendrecv_HandlesZeroCount)
    {
        if (!hardware_available_)
        {
            GTEST_SKIP() << "RCCL hardware not available";
        }

        auto group = createSingleROCmGroup();
        if (!backend_->initialize(group))
        {
            GTEST_SKIP() << "Backend initialization failed";
        }

        float sendbuf[1];
        float recvbuf[1];
        // Zero count should either succeed (no-op) or fail gracefully
        bool result = backend_->sendrecv(sendbuf, recvbuf, 0, CollectiveDataType::FLOAT32, 0);
        (void)result;
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // P2P Operations - Invalid Peer Tests
    // ═══════════════════════════════════════════════════════════════════════════

    TEST_F(Test__RCCLBackend, Send_FailsWithInvalidPeer)
    {
        if (!hardware_available_)
        {
            GTEST_SKIP() << "RCCL hardware not available";
        }

        auto group = createSingleROCmGroup();
        if (!backend_->initialize(group))
        {
            GTEST_SKIP() << "Backend initialization failed";
        }

        float buffer[64];
        // Peer rank 99 is invalid for a single-device group
        EXPECT_FALSE(backend_->send(buffer, 64, CollectiveDataType::FLOAT32, 99, 0))
            << "send() should fail with invalid peer rank";
    }

    TEST_F(Test__RCCLBackend, Recv_FailsWithInvalidPeer)
    {
        if (!hardware_available_)
        {
            GTEST_SKIP() << "RCCL hardware not available";
        }

        auto group = createSingleROCmGroup();
        if (!backend_->initialize(group))
        {
            GTEST_SKIP() << "Backend initialization failed";
        }

        float buffer[64];
        // Peer rank -1 is invalid
        EXPECT_FALSE(backend_->recv(buffer, 64, CollectiveDataType::FLOAT32, -1, 0))
            << "recv() should fail with negative peer rank";
    }

    TEST_F(Test__RCCLBackend, Sendrecv_FailsWithInvalidPeer)
    {
        if (!hardware_available_)
        {
            GTEST_SKIP() << "RCCL hardware not available";
        }

        auto group = createSingleROCmGroup();
        if (!backend_->initialize(group))
        {
            GTEST_SKIP() << "Backend initialization failed";
        }

        float sendbuf[64];
        float recvbuf[64];
        // Peer rank 100 is invalid
        EXPECT_FALSE(backend_->sendrecv(sendbuf, recvbuf, 64, CollectiveDataType::FLOAT32, 100))
            << "sendrecv() should fail with invalid peer rank";
    }

} // namespace llaminar2::test

#else // !HAVE_ROCM

// Stub test when ROCm not available
TEST(Test__RCCLBackend, RequiresROCm)
{
    GTEST_SKIP() << "RCCLBackend requires HAVE_ROCM";
}

#endif // HAVE_ROCM
