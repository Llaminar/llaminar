/**
 * @file Test__TensorFactory_BARBacked.cpp
 * @brief Unit tests for TensorFactory BAR-backed tensor support
 *
 * Tests the following TensorFactory methods:
 * - setDirectP2P(): Sets DirectP2PEngine context for BAR allocation
 * - canCreateBARBacked(): Checks if BAR-backed tensors are available
 * - createFP32BARBacked(): Creates FP32 tensor in BAR memory
 *
 * These tests verify the API contract and error handling. Full integration
 * tests with actual GPU hardware are in Test__PCIeBARBackendIntegration.cpp.
 */

#include <gtest/gtest.h>
#include "tensors/TensorFactory.h"
#include "tensors/TensorClasses.h"
#include "backends/DeviceId.h"
#include "utils/MPIContext.h"

#if defined(HAVE_CUDA) && defined(HAVE_ROCM)
#include "backends/p2p/DirectP2P.h"
#endif

namespace llaminar2
{
    namespace test
    {

        class Test__TensorFactory_BARBacked : public ::testing::Test
        {
        protected:
            std::unique_ptr<TensorFactory> factory_;
            std::unique_ptr<MPIContext> mpi_ctx_;

            void SetUp() override
            {
                // Create a mock MPI context (single rank with rank=0, world_size=1)
                mpi_ctx_ = std::make_unique<MPIContext>(0, 1);
                factory_ = std::make_unique<TensorFactory>(*mpi_ctx_);
            }
        };

        // =========================================================================
        // Test: SetDirectP2P - Factory accepts and stores P2P context
        // =========================================================================

        TEST_F(Test__TensorFactory_BARBacked, SetDirectP2P)
        {
            // Before setting P2P, canCreateBARBacked should return false
            EXPECT_FALSE(factory_->canCreateBARBacked());

#if defined(HAVE_CUDA) && defined(HAVE_ROCM)
            // Create a DirectP2PEngine (not initialized, so BAR not active)
            auto p2p = std::make_shared<DirectP2PEngine>();

            // Set the P2P context
            factory_->setDirectP2P(p2p);

            // Still false because BAR is not active (not initialized)
            EXPECT_FALSE(factory_->canCreateBARBacked());

            // Clear the P2P context
            factory_->setDirectP2P(nullptr);
            EXPECT_FALSE(factory_->canCreateBARBacked());
#else
            // Without both CUDA and ROCm, we can only test that setting nullptr works
            factory_->setDirectP2P(nullptr);
            EXPECT_FALSE(factory_->canCreateBARBacked());
            GTEST_LOG_(INFO) << "Full DirectP2P test requires HAVE_CUDA and HAVE_ROCM";
#endif
        }

        // =========================================================================
        // Test: CanCreateBARBacked_True - Returns true when P2P set with valid BAR
        // =========================================================================

        TEST_F(Test__TensorFactory_BARBacked, CanCreateBARBacked_True)
        {
#if defined(HAVE_CUDA) && defined(HAVE_ROCM)
            // This test requires actual hardware with initialized BAR
            // We'll check if the system supports it first
            auto caps = DirectP2PEngine::probeCapabilities();
            if (!caps.canDoDirectP2P())
            {
                GTEST_SKIP() << "PCIe BAR P2P not available: " << caps.describe();
            }

            // Create and initialize DirectP2PEngine with actual hardware
            auto p2p = std::make_shared<DirectP2PEngine>();

            // Try to initialize with first available CUDA and ROCm devices
            bool initialized = p2p->initializePCIeBar(
                DeviceId::cuda(0),
                DeviceId::rocm(0));

            if (!initialized)
            {
                GTEST_SKIP() << "Failed to initialize PCIe BAR P2P (requires CAP_SYS_ADMIN)";
            }

            // Set the P2P context
            factory_->setDirectP2P(p2p);

            // Now canCreateBARBacked should return true
            EXPECT_TRUE(factory_->canCreateBARBacked());

            GTEST_LOG_(INFO) << "BAR mapped size: " << p2p->getBarMappedSize() << " bytes";
#else
            GTEST_SKIP() << "Test requires both HAVE_CUDA and HAVE_ROCM";
#endif
        }

        // =========================================================================
        // Test: CanCreateBARBacked_False_NoP2P - Returns false when P2P not set
        // =========================================================================

        TEST_F(Test__TensorFactory_BARBacked, CanCreateBARBacked_False_NoP2P)
        {
            // Without setting DirectP2P, canCreateBARBacked should return false
            EXPECT_FALSE(factory_->canCreateBARBacked());
        }

        // =========================================================================
        // Test: CreateFP32BARBacked_Shape - Created tensor has correct shape
        // =========================================================================

        TEST_F(Test__TensorFactory_BARBacked, CreateFP32BARBacked_Shape)
        {
#if defined(HAVE_CUDA) && defined(HAVE_ROCM)
            auto caps = DirectP2PEngine::probeCapabilities();
            if (!caps.canDoDirectP2P())
            {
                GTEST_SKIP() << "PCIe BAR P2P not available";
            }

            auto p2p = std::make_shared<DirectP2PEngine>();
            if (!p2p->initializePCIeBar(DeviceId::cuda(0), DeviceId::rocm(0)))
            {
                GTEST_SKIP() << "Failed to initialize PCIe BAR P2P";
            }

            factory_->setDirectP2P(p2p);
            ASSERT_TRUE(factory_->canCreateBARBacked());

            // Create a BAR-backed tensor with specific shape
            std::vector<size_t> shape = {32, 64};
            auto tensor = factory_->createFP32BARBacked(shape, DeviceId::rocm(0), DeviceId::cuda(0));

            ASSERT_NE(tensor, nullptr);
            ASSERT_EQ(tensor->shape().size(), 2);
            EXPECT_EQ(tensor->shape()[0], 32);
            EXPECT_EQ(tensor->shape()[1], 64);
#else
            GTEST_SKIP() << "Test requires both HAVE_CUDA and HAVE_ROCM";
#endif
        }

        // =========================================================================
        // Test: CreateFP32BARBacked_IsBARBacked - isBARBacked() returns true
        // =========================================================================

        TEST_F(Test__TensorFactory_BARBacked, CreateFP32BARBacked_IsBARBacked)
        {
#if defined(HAVE_CUDA) && defined(HAVE_ROCM)
            auto caps = DirectP2PEngine::probeCapabilities();
            if (!caps.canDoDirectP2P())
            {
                GTEST_SKIP() << "PCIe BAR P2P not available";
            }

            auto p2p = std::make_shared<DirectP2PEngine>();
            if (!p2p->initializePCIeBar(DeviceId::cuda(0), DeviceId::rocm(0)))
            {
                GTEST_SKIP() << "Failed to initialize PCIe BAR P2P";
            }

            factory_->setDirectP2P(p2p);

            auto tensor = factory_->createFP32BARBacked({32, 64}, DeviceId::rocm(0), DeviceId::cuda(0));

            ASSERT_NE(tensor, nullptr);
            EXPECT_TRUE(tensor->isBARBacked());

            // Regular FP32 tensor should NOT be BAR-backed
            auto regular_tensor = factory_->createFP32({32, 64});
            EXPECT_FALSE(regular_tensor->isBARBacked());
#else
            GTEST_SKIP() << "Test requires both HAVE_CUDA and HAVE_ROCM";
#endif
        }

        // =========================================================================
        // Test: CreateFP32BARBacked_DeviceAffinity - Reports ROCm device affinity
        // =========================================================================

        TEST_F(Test__TensorFactory_BARBacked, CreateFP32BARBacked_DeviceAffinity)
        {
#if defined(HAVE_CUDA) && defined(HAVE_ROCM)
            auto caps = DirectP2PEngine::probeCapabilities();
            if (!caps.canDoDirectP2P())
            {
                GTEST_SKIP() << "PCIe BAR P2P not available";
            }

            auto p2p = std::make_shared<DirectP2PEngine>();
            if (!p2p->initializePCIeBar(DeviceId::cuda(0), DeviceId::rocm(0)))
            {
                GTEST_SKIP() << "Failed to initialize PCIe BAR P2P";
            }

            factory_->setDirectP2P(p2p);

            // Create with ROCm device 0 as host
            auto tensor = factory_->createFP32BARBacked({32, 64}, DeviceId::rocm(0), DeviceId::cuda(0));

            ASSERT_NE(tensor, nullptr);

            // home_device should be the ROCm device (where the BAR memory resides)
            EXPECT_TRUE(tensor->home_device().is_rocm());
            EXPECT_EQ(tensor->home_device().ordinal, 0);  // ordinal is a member, not method
#else
            GTEST_SKIP() << "Test requires both HAVE_CUDA and HAVE_ROCM";
#endif
        }

        // =========================================================================
        // Test: CreateFP32BARBacked_DualPointers - Both pointers are valid
        // =========================================================================

        TEST_F(Test__TensorFactory_BARBacked, CreateFP32BARBacked_DualPointers)
        {
#if defined(HAVE_CUDA) && defined(HAVE_ROCM)
            auto caps = DirectP2PEngine::probeCapabilities();
            if (!caps.canDoDirectP2P())
            {
                GTEST_SKIP() << "PCIe BAR P2P not available";
            }

            auto p2p = std::make_shared<DirectP2PEngine>();
            if (!p2p->initializePCIeBar(DeviceId::cuda(0), DeviceId::rocm(0)))
            {
                GTEST_SKIP() << "Failed to initialize PCIe BAR P2P";
            }

            factory_->setDirectP2P(p2p);

            auto tensor = factory_->createFP32BARBacked({32, 64}, DeviceId::rocm(0), DeviceId::cuda(0));

            ASSERT_NE(tensor, nullptr);

            // Both ROCm and CUDA pointers should be valid (non-null)
            void *rocm_ptr = tensor->rocm_data_ptr();
            void *cuda_ptr = tensor->gpu_data_ptr();

            EXPECT_NE(rocm_ptr, nullptr) << "ROCm pointer should be valid";
            EXPECT_NE(cuda_ptr, nullptr) << "CUDA pointer should be valid";

            // The pointers should be different (one is mmap'd BAR, other is CUDA device ptr)
            // But they point to the same physical memory
            GTEST_LOG_(INFO) << "ROCm ptr: " << rocm_ptr;
            GTEST_LOG_(INFO) << "CUDA ptr: " << cuda_ptr;
#else
            GTEST_SKIP() << "Test requires both HAVE_CUDA and HAVE_ROCM";
#endif
        }

        // =========================================================================
        // Test: CreateFP32BARBacked_ThrowsWithoutP2P - Throws when P2P not configured
        // =========================================================================

        TEST_F(Test__TensorFactory_BARBacked, CreateFP32BARBacked_ThrowsWithoutP2P)
        {
#if defined(HAVE_CUDA) && defined(HAVE_ROCM)
            // Do NOT set DirectP2P
            EXPECT_FALSE(factory_->canCreateBARBacked());

            // Attempting to create BAR-backed tensor should throw
            EXPECT_THROW(
                factory_->createFP32BARBacked({32, 64}, DeviceId::rocm(0), DeviceId::cuda(0)),
                std::runtime_error);
#else
            // Without both backends, this should also throw
            EXPECT_THROW(
                factory_->createFP32BARBacked({32, 64}, DeviceId::rocm(0), DeviceId::cuda(0)),
                std::runtime_error);
#endif
        }

        // =========================================================================
        // Test: CreateFP32BARBacked_ValidatesDeviceTypes - Wrong device types fail
        // =========================================================================

        TEST_F(Test__TensorFactory_BARBacked, CreateFP32BARBacked_ValidatesDeviceTypes)
        {
#if defined(HAVE_CUDA) && defined(HAVE_ROCM)
            auto caps = DirectP2PEngine::probeCapabilities();
            if (!caps.canDoDirectP2P())
            {
                GTEST_SKIP() << "PCIe BAR P2P not available";
            }

            auto p2p = std::make_shared<DirectP2PEngine>();
            if (!p2p->initializePCIeBar(DeviceId::cuda(0), DeviceId::rocm(0)))
            {
                GTEST_SKIP() << "Failed to initialize PCIe BAR P2P";
            }

            factory_->setDirectP2P(p2p);

            // Wrong device type for rocm_device (passing CUDA)
            auto tensor1 = factory_->createFP32BARBacked({32, 64}, DeviceId::cuda(0), DeviceId::cuda(0));
            EXPECT_EQ(tensor1, nullptr);

            // Wrong device type for cuda_device (passing ROCm)
            auto tensor2 = factory_->createFP32BARBacked({32, 64}, DeviceId::rocm(0), DeviceId::rocm(0));
            EXPECT_EQ(tensor2, nullptr);
#else
            GTEST_SKIP() << "Test requires both HAVE_CUDA and HAVE_ROCM";
#endif
        }

    } // namespace test
} // namespace llaminar2
