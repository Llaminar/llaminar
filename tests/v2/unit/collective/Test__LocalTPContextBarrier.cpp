/**
 * @file Test__LocalTPContextBarrier.cpp
 * @brief Unit tests for LocalTPContext barrier synchronization
 *
 * Tests the NCCL-style barrier mechanism for PCIeBAR heterogeneous allreduce
 * where multiple device threads must rendezvous before the actual data transfer.
 *
 * NOTE: The barrier mechanism is only active for PCIeBAR backend with CUDA+ROCm.
 * Single-device tests verify basic functionality, while PCIeBAR tests verify
 * the actual barrier synchronization.
 *
 * @author David Sanftenberg
 * @date January 2026
 */

#include <gtest/gtest.h>
#include <thread>
#include <atomic>
#include <chrono>
#include <vector>
#include <functional>
#include <condition_variable>
#include <mutex>

#include "collective/LocalTPContext.h"
#include "collective/ICollectiveBackend.h"
#include "backends/GlobalDeviceAddress.h"
#include "tensors/Tensors.h"
#include "../../utils/TestTensorFactory.h"

using namespace llaminar2;
using namespace llaminar2::test;

// =============================================================================
// Test Fixture
// =============================================================================

/**
 * @class Test__LocalTPContextBarrier
 * @brief Test fixture for LocalTPContext barrier synchronization
 *
 * Tests the barrier mechanism using realistic device configurations.
 * - Single-device tests verify no-op behavior
 * - PCIeBAR tests verify actual barrier rendezvous (requires CUDA+ROCm)
 */
class Test__LocalTPContextBarrier : public ::testing::Test
{
protected:
    void SetUp() override
    {
        // Create test devices - use CUDA + ROCm for realistic heterogeneous setup
        cuda0_ = GlobalDeviceAddress::cuda(0, 0);
        rocm0_ = GlobalDeviceAddress::rocm(0, 0);
        cpu0_ = GlobalDeviceAddress::cpu(0);
    }

    void TearDown() override
    {
        // Ensure all threads are joined
    }

    GlobalDeviceAddress cuda0_;
    GlobalDeviceAddress rocm0_;
    GlobalDeviceAddress cpu0_;
};

// =============================================================================
// Single Device Tests (No Barrier Needed)
// =============================================================================

/**
 * @test Single device allreduce should be a no-op (no barrier needed)
 */
TEST_F(Test__LocalTPContextBarrier, SingleDeviceNoBarrier)
{
    auto ctx = createLocalTPContext({cuda0_}, {}, CollectiveBackendType::AUTO);
    ASSERT_NE(ctx, nullptr);
    EXPECT_EQ(ctx->degree(), 1);

    auto tensor = TestTensorFactory::createFP32({1024});
    float *data = tensor->mutable_data();
    for (size_t i = 0; i < tensor->numel(); ++i)
    {
        data[i] = static_cast<float>(i);
    }

    // Single device - should return immediately with no barrier
    bool result = ctx->allreduce(tensor.get());
    EXPECT_TRUE(result);
}

/**
 * @test Single CPU device allreduce should work
 */
TEST_F(Test__LocalTPContextBarrier, SingleCPUDevice)
{
    auto ctx = createLocalTPContext({cpu0_}, {}, CollectiveBackendType::HOST);
    ASSERT_NE(ctx, nullptr);
    EXPECT_EQ(ctx->degree(), 1);

    auto tensor = TestTensorFactory::createFP32({256});
    float *data = tensor->mutable_data();
    for (size_t i = 0; i < tensor->numel(); ++i)
    {
        data[i] = 1.0f;
    }

    // Single device - should return immediately
    bool result = ctx->allreduce(tensor.get());
    EXPECT_TRUE(result);
}

/**
 * @test Verify context state initialization
 */
TEST_F(Test__LocalTPContextBarrier, ContextStateInitialization)
{
    auto ctx = createLocalTPContext({cuda0_}, {}, CollectiveBackendType::AUTO);
    ASSERT_NE(ctx, nullptr);

    // Context should be created without errors
    EXPECT_EQ(ctx->degree(), 1);
}

// =============================================================================
// Edge Case Tests
// =============================================================================

/**
 * @test Null tensor should return false
 */
TEST_F(Test__LocalTPContextBarrier, NullTensorReturnsError)
{
    auto ctx = createLocalTPContext({cuda0_}, {}, CollectiveBackendType::AUTO);
    ASSERT_NE(ctx, nullptr);

    bool result = ctx->allreduce(nullptr);
    EXPECT_FALSE(result);
}

/**
 * @test Rapid sequential allreduces from single device context
 */
TEST_F(Test__LocalTPContextBarrier, RapidSequentialAllreducesSingleDevice)
{
    auto ctx = createLocalTPContext({cuda0_}, {}, CollectiveBackendType::AUTO);
    ASSERT_NE(ctx, nullptr);

    const int num_iterations = 100;
    auto tensor = TestTensorFactory::createFP32({512});

    for (size_t i = 0; i < tensor->numel(); ++i)
    {
        tensor->mutable_data()[i] = static_cast<float>(i);
    }

    for (int i = 0; i < num_iterations; ++i)
    {
        bool result = ctx->allreduce(tensor.get());
        EXPECT_TRUE(result) << "Failed at iteration " << i;
    }
}

/**
 * @test Out-of-place allreduce with single device
 */
TEST_F(Test__LocalTPContextBarrier, OutOfPlaceAllreduceSingleDevice)
{
    auto ctx = createLocalTPContext({cuda0_}, {}, CollectiveBackendType::AUTO);
    ASSERT_NE(ctx, nullptr);

    auto input = TestTensorFactory::createFP32({256});
    auto output = TestTensorFactory::createFP32({256});

    for (size_t i = 0; i < input->numel(); ++i)
    {
        input->mutable_data()[i] = static_cast<float>(i);
        output->mutable_data()[i] = 0.0f;
    }

    bool result = ctx->allreduce(input.get(), output.get());
    EXPECT_TRUE(result);

    // Output should have been populated (copied from input for single device)
    bool has_nonzero = false;
    for (size_t i = 0; i < output->numel(); ++i)
    {
        if (output->data()[i] != 0.0f)
        {
            has_nonzero = true;
            break;
        }
    }
    EXPECT_TRUE(has_nonzero);
}

// =============================================================================
// PCIeBAR-Specific Tests - MIGRATED TO INTEGRATION TESTS
// =============================================================================
// NOTE: Hardware-dependent PCIeBAR barrier tests (PCIeBarBarrierRendezvous,
// PCIeBarMultipleBarrierCycles, PCIeBarStressTest, FirstArrivalWaitsSecondTriggers)
// have been migrated to integration tests in Test__LocalTPBackendBehavior.cpp.
// Those tests require actual CUDA+ROCm hardware with PCIe BAR support.

// =============================================================================
// Backend Fallback Tests
// =============================================================================

/**
 * @test Verify backend auto-detection doesn't affect single-device behavior
 */
TEST_F(Test__LocalTPContextBarrier, AutoDetectedBackendWorks)
{
    // AUTO should work for single device
    auto ctx = createLocalTPContext({cpu0_}, {}, CollectiveBackendType::AUTO);
    ASSERT_NE(ctx, nullptr);

    auto tensor = TestTensorFactory::createFP32({256});
    for (size_t i = 0; i < tensor->numel(); ++i)
    {
        tensor->mutable_data()[i] = static_cast<float>(i);
    }

    bool result = ctx->allreduce(tensor.get());
    EXPECT_TRUE(result);
}
