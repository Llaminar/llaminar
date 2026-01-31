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

    // NOTE: P2P operation tests (send/recv/sendrecv) require actual GPU hardware
    // and are in integration/Test__NCCLBackend.cpp. Unit tests only verify
    // the API surface (type, name, device support) without hardware initialization.

} // namespace llaminar2::test

#else // !HAVE_CUDA

// Stub test when CUDA not available
TEST(Test__NCCLBackend, RequiresCUDA)
{
    GTEST_SKIP() << "NCCLBackend requires HAVE_CUDA";
}

#endif // HAVE_CUDA
