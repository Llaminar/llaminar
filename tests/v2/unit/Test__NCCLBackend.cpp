/**
 * @file Test__NCCLBackend.cpp
 * @brief Unit tests for NCCLBackend P2P operations
 *
 * Tests the point-to-point send/recv/sendrecv primitives for NCCL backend.
 * Validates error handling for uninitialized state, invalid parameters,
 * and edge cases.
 *
 * @note Full integration tests with actual NCCL operations are in
 *       integration/Test__CrossBackendP2P.cpp
 *
 * @author David Sanftenberg
 * @date January 2026
 */

#include <gtest/gtest.h>
#include "v2/collective/backends/NCCLBackend.h"
#include "v2/collective/DeviceGroup.h"
#include "v2/backends/DeviceId.h"

#ifdef HAVE_CUDA

namespace llaminar2::test
{

    // ═══════════════════════════════════════════════════════════════════════════
    // Test Fixture
    // ═══════════════════════════════════════════════════════════════════════════

    class Test__NCCLBackend : public ::testing::Test
    {
    protected:
        void SetUp() override
        {
            backend_ = std::make_unique<NCCLBackend>();
            hardware_available_ = backend_->isAvailable();
        }

        void TearDown() override
        {
            if (backend_ && backend_->isInitialized())
            {
                backend_->shutdown();
            }
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

        // Helper to create a 2-device CUDA group (for P2P tests)
        DeviceGroup createTwoCUDAGroup()
        {
            DeviceGroupBuilder builder;
            return builder
                .setName("two_cuda")
                .setScope(CollectiveScope::LOCAL)
                .addDevice(DeviceId::cuda(0))
                .addDevice(DeviceId::cuda(1))
                .setLocalRank(0)
                .build();
        }

        std::unique_ptr<NCCLBackend> backend_;
        bool hardware_available_ = false;
    };

    // ═══════════════════════════════════════════════════════════════════════════
    // Identity Tests
    // ═══════════════════════════════════════════════════════════════════════════

    TEST_F(Test__NCCLBackend, TypeIsNCCL)
    {
        EXPECT_EQ(backend_->type(), CollectiveBackendType::NCCL);
    }

    TEST_F(Test__NCCLBackend, NameIsNCCL)
    {
        EXPECT_EQ(backend_->name(), "NCCL");
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // Capability Tests
    // ═══════════════════════════════════════════════════════════════════════════

    TEST_F(Test__NCCLBackend, SupportsCUDADevices)
    {
        EXPECT_TRUE(backend_->supportsDevice(DeviceType::CUDA));
    }

    TEST_F(Test__NCCLBackend, DoesNotSupportROCm)
    {
        EXPECT_FALSE(backend_->supportsDevice(DeviceType::ROCm));
    }

    TEST_F(Test__NCCLBackend, DoesNotSupportCPU)
    {
        EXPECT_FALSE(backend_->supportsDevice(DeviceType::CPU));
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // P2P Operations - Not Initialized Tests
    // ═══════════════════════════════════════════════════════════════════════════

    TEST_F(Test__NCCLBackend, Send_FailsWhenNotInitialized)
    {
        if (!hardware_available_)
        {
            GTEST_SKIP() << "NCCL hardware not available";
        }

        float buffer[64];
        EXPECT_FALSE(backend_->send(buffer, 64, CollectiveDataType::FLOAT32, 1, 0))
            << "send() should fail when backend is not initialized";
    }

    TEST_F(Test__NCCLBackend, Recv_FailsWhenNotInitialized)
    {
        if (!hardware_available_)
        {
            GTEST_SKIP() << "NCCL hardware not available";
        }

        float buffer[64];
        EXPECT_FALSE(backend_->recv(buffer, 64, CollectiveDataType::FLOAT32, 0, 0))
            << "recv() should fail when backend is not initialized";
    }

    TEST_F(Test__NCCLBackend, Sendrecv_FailsWhenNotInitialized)
    {
        if (!hardware_available_)
        {
            GTEST_SKIP() << "NCCL hardware not available";
        }

        float sendbuf[64];
        float recvbuf[64];
        EXPECT_FALSE(backend_->sendrecv(sendbuf, recvbuf, 64, CollectiveDataType::FLOAT32, 1))
            << "sendrecv() should fail when backend is not initialized";
    }

    // ═══════════════════════════════════════════════════════════════════════════
    // P2P Operations - Null Buffer Tests
    // ═══════════════════════════════════════════════════════════════════════════

    TEST_F(Test__NCCLBackend, Send_FailsWithNullBuffer)
    {
        if (!hardware_available_)
        {
            GTEST_SKIP() << "NCCL hardware not available";
        }

        auto group = createSingleCUDAGroup();
        if (!backend_->initialize(group))
        {
            GTEST_SKIP() << "Backend initialization failed";
        }

        EXPECT_FALSE(backend_->send(nullptr, 64, CollectiveDataType::FLOAT32, 0, 0))
            << "send() should fail with null buffer";
    }

    TEST_F(Test__NCCLBackend, Recv_FailsWithNullBuffer)
    {
        if (!hardware_available_)
        {
            GTEST_SKIP() << "NCCL hardware not available";
        }

        auto group = createSingleCUDAGroup();
        if (!backend_->initialize(group))
        {
            GTEST_SKIP() << "Backend initialization failed";
        }

        EXPECT_FALSE(backend_->recv(nullptr, 64, CollectiveDataType::FLOAT32, 0, 0))
            << "recv() should fail with null buffer";
    }

    TEST_F(Test__NCCLBackend, Sendrecv_FailsWithNullSendBuffer)
    {
        if (!hardware_available_)
        {
            GTEST_SKIP() << "NCCL hardware not available";
        }

        auto group = createSingleCUDAGroup();
        if (!backend_->initialize(group))
        {
            GTEST_SKIP() << "Backend initialization failed";
        }

        float recvbuf[64];
        EXPECT_FALSE(backend_->sendrecv(nullptr, recvbuf, 64, CollectiveDataType::FLOAT32, 0))
            << "sendrecv() should fail with null send buffer";
    }

    TEST_F(Test__NCCLBackend, Sendrecv_FailsWithNullRecvBuffer)
    {
        if (!hardware_available_)
        {
            GTEST_SKIP() << "NCCL hardware not available";
        }

        auto group = createSingleCUDAGroup();
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

    TEST_F(Test__NCCLBackend, Send_HandlesZeroCount)
    {
        if (!hardware_available_)
        {
            GTEST_SKIP() << "NCCL hardware not available";
        }

        auto group = createSingleCUDAGroup();
        if (!backend_->initialize(group))
        {
            GTEST_SKIP() << "Backend initialization failed";
        }

        float buffer[1];
        // Zero count should either succeed (no-op) or fail gracefully
        // The exact behavior depends on NCCL implementation
        bool result = backend_->send(buffer, 0, CollectiveDataType::FLOAT32, 0, 0);
        // We just verify no crash occurs - result can be true (no-op success) or false (rejected)
        (void)result;
    }

    TEST_F(Test__NCCLBackend, Recv_HandlesZeroCount)
    {
        if (!hardware_available_)
        {
            GTEST_SKIP() << "NCCL hardware not available";
        }

        auto group = createSingleCUDAGroup();
        if (!backend_->initialize(group))
        {
            GTEST_SKIP() << "Backend initialization failed";
        }

        float buffer[1];
        // Zero count should either succeed (no-op) or fail gracefully
        bool result = backend_->recv(buffer, 0, CollectiveDataType::FLOAT32, 0, 0);
        (void)result;
    }

    TEST_F(Test__NCCLBackend, Sendrecv_HandlesZeroCount)
    {
        if (!hardware_available_)
        {
            GTEST_SKIP() << "NCCL hardware not available";
        }

        auto group = createSingleCUDAGroup();
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

    TEST_F(Test__NCCLBackend, Send_FailsWithInvalidPeer)
    {
        if (!hardware_available_)
        {
            GTEST_SKIP() << "NCCL hardware not available";
        }

        auto group = createSingleCUDAGroup();
        if (!backend_->initialize(group))
        {
            GTEST_SKIP() << "Backend initialization failed";
        }

        float buffer[64];
        // Peer rank 99 is invalid for a single-device group
        EXPECT_FALSE(backend_->send(buffer, 64, CollectiveDataType::FLOAT32, 99, 0))
            << "send() should fail with invalid peer rank";
    }

    TEST_F(Test__NCCLBackend, Recv_FailsWithInvalidPeer)
    {
        if (!hardware_available_)
        {
            GTEST_SKIP() << "NCCL hardware not available";
        }

        auto group = createSingleCUDAGroup();
        if (!backend_->initialize(group))
        {
            GTEST_SKIP() << "Backend initialization failed";
        }

        float buffer[64];
        // Peer rank -1 is invalid
        EXPECT_FALSE(backend_->recv(buffer, 64, CollectiveDataType::FLOAT32, -1, 0))
            << "recv() should fail with negative peer rank";
    }

    TEST_F(Test__NCCLBackend, Sendrecv_FailsWithInvalidPeer)
    {
        if (!hardware_available_)
        {
            GTEST_SKIP() << "NCCL hardware not available";
        }

        auto group = createSingleCUDAGroup();
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

#else // !HAVE_CUDA

// Stub test when CUDA not available
TEST(Test__NCCLBackend, RequiresCUDA)
{
    GTEST_SKIP() << "NCCLBackend requires HAVE_CUDA";
}

#endif // HAVE_CUDA
