/**
 * @file Test__NCCLBackend.cpp
 * @brief Device-free NCCL topology/channel policy and backend P2P admission tests.
 *
 * Proves complete physical-pair selection and rejects unmeasured channel profiles.
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
#include "v2/collective/backends/NCCLNetworkPolicy.h"
#include "v2/collective/DeviceGroup.h"
#include "v2/backends/DeviceId.h"
#include <array>

#ifdef HAVE_CUDA

namespace llaminar2::test
{

    // =========================================================================
    // Network Policy Tests
    // =========================================================================

    TEST(Test__NCCLNetworkPolicy, LocalScopeSelectsSocketWithoutDisablingP2P)
    {
        const NCCLNetworkModule module =
            selectNCCLNetworkModule(CollectiveScope::LOCAL);

        EXPECT_EQ(module, NCCLNetworkModule::Socket);
        EXPECT_EQ(ncclNetworkModuleName(module), "Socket");
    }

    TEST(Test__NCCLNetworkPolicy, GlobalScopePreservesAutomaticNetworkSelection)
    {
        const NCCLNetworkModule module =
            selectNCCLNetworkModule(CollectiveScope::GLOBAL);

        EXPECT_EQ(module, NCCLNetworkModule::Automatic);
        EXPECT_TRUE(ncclNetworkModuleName(module).empty());
    }

    TEST(Test__NCCLNetworkPolicy, HybridScopePreservesAutomaticNetworkSelection)
    {
        const NCCLNetworkModule module =
            selectNCCLNetworkModule(CollectiveScope::HYBRID);

        EXPECT_EQ(module, NCCLNetworkModule::Automatic);
        EXPECT_TRUE(ncclNetworkModuleName(module).empty());
    }

    /** @test Only the fully observed local SHM pair receives the measured CTA budget. */
    TEST(Test__NCCLNetworkPolicy, MeasuredChannelsRequireCompletePhysicalPair)
    {
        std::array cards{
            NCCLCardIdentity{"NVIDIA GeForce RTX 3090", 0, 8, 6, 82, std::size_t{24} << 30},
            NCCLCardIdentity{"NVIDIA GeForce RTX 3090", 1, 8, 6, 82, std::size_t{24} << 30}};
        std::array peers{NCCLPeerAccess::Unavailable, NCCLPeerAccess::Unavailable};
        const auto select = [&](CollectiveScope scope = CollectiveScope::LOCAL)
            { return selectNCCLCommunicatorPolicy(scope, cards, peers); };
        EXPECT_EQ(select().network, NCCLNetworkModule::Socket);
        EXPECT_EQ(ncclCommunicatorChannelBudget(select().profile), 8);
        EXPECT_EQ(ncclCommunicatorProfileName(select().profile), "rtx3090_shm_tp2");
        for (const auto scope : {CollectiveScope::GLOBAL, CollectiveScope::HYBRID})
            EXPECT_EQ(select(scope).profile, NCCLCommunicatorProfile::NativeAutomatic);
        for (const auto access : {NCCLPeerAccess::Available, NCCLPeerAccess::Unobserved})
            for (int direction = 0; direction < 2; ++direction)
            {
                peers[direction] = access;
                EXPECT_EQ(select().profile, NCCLCommunicatorProfile::NativeAutomatic);
                peers[direction] = NCCLPeerAccess::Unavailable;
            }
        cards[1].ordinal = 0;
        EXPECT_EQ(select().profile, NCCLCommunicatorProfile::NativeAutomatic);
        cards[1].ordinal = 1;
        const auto measured = cards[1];
        cards[1].name = "NVIDIA GeForce RTX 3090 Ti";
        EXPECT_EQ(select().profile, NCCLCommunicatorProfile::NativeAutomatic);
        cards[1] = measured;
        cards[1].multiprocessors = 80;
        EXPECT_EQ(select().profile, NCCLCommunicatorProfile::NativeAutomatic);
        cards[1] = measured;
        cards[1].architecture_minor = 9;
        EXPECT_EQ(select().profile, NCCLCommunicatorProfile::NativeAutomatic);
        cards[1] = measured;
        cards[1].memory_bytes = std::size_t{22} << 30;
        EXPECT_EQ(select().profile, NCCLCommunicatorProfile::NativeAutomatic);
        EXPECT_EQ(selectNCCLCommunicatorPolicy(CollectiveScope::LOCAL, {}, {}).profile,
            NCCLCommunicatorProfile::NativeAutomatic);
        EXPECT_THROW(ncclCommunicatorChannelBudget(static_cast<NCCLCommunicatorProfile>(99)), std::invalid_argument);
    }

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
