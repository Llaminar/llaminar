/**
 * @file Test__ITPContext_LocalTP.cpp
 * @brief Integration tests for ITPContext with LocalTPContext implementation
 * @author David Sanftenberg
 * @date February 2026
 *
 * Tests that the ITPContext interface can be used correctly when backed by
 * LocalTPContext. These tests focus on:
 *
 * 1. Interface compliance - degree(), myIndex(), isLocal() through ITPContext*
 * 2. Polymorphic usage - using LocalTPContext through base ITPContext pointer
 * 3. Device index management - setCurrentDeviceIndex() and myIndex() interaction
 *
 * Note on HOST backend:
 * ---------------------
 * The HOST backend is designed for heterogeneous GPU scenarios (CUDA + ROCm)
 * where data is staged through host memory. When used with pure CPU devices,
 * LocalTPContext's collective operations are not fully supported because:
 * - LocalTPContext expects GPU buffers for tensor->ensureOnDevice()
 * - The code paths assume GPU-based tensor management
 *
 * For actual collective operation testing with real data, see:
 * - Test__CollectiveBackendIntegration.cpp (MPI-based CPU collectives)
 * - Test__LocalTPMultiDevice.cpp (GPU-based NCCL/RCCL collectives)
 * - Test__LocalTPBackendBehavior.cpp (PCIeBAR heterogeneous collectives)
 *
 * These tests verify the ITPContext interface abstraction works correctly,
 * which is the foundation for the unified TP interface plan.
 */

#include <gtest/gtest.h>
#include <cmath>
#include <numeric>

#include "collective/ITPContext.h"
#include "collective/ILocalTPContext.h"
#include "collective/LocalTPContext.h"
#include "backends/GlobalDeviceAddress.h"
#include "config/OrchestrationConfig.h"
#include "tensors/TensorClasses.h"
#include "../../utils/TestTensorFactory.h"

using namespace llaminar2;
using namespace llaminar2::test;

// =============================================================================
// Test Fixture
// =============================================================================

class Test__ITPContext_LocalTP : public ::testing::Test
{
protected:
    void SetUp() override
    {
        cpu0_ = GlobalDeviceAddress::cpu(0);
        cpu1_ = GlobalDeviceAddress::cpu(1);

        // Create a LocalTPContext with HOST backend (works without GPU)
        local_ctx_ = createLocalTPContext(
            {cpu0_, cpu1_},
            {},  // Equal weights
            CollectiveBackendType::HOST);

        // Cast to base interface for testing
        base_ctx_ = local_ctx_.get();

        // Also keep a pointer to the concrete class for setCurrentDeviceIndex()
        concrete_ctx_ = dynamic_cast<LocalTPContext *>(local_ctx_.get());
        ASSERT_NE(concrete_ctx_, nullptr) << "Failed to cast to LocalTPContext";
    }

    GlobalDeviceAddress cpu0_;
    GlobalDeviceAddress cpu1_;
    std::unique_ptr<ILocalTPContext> local_ctx_;
    ITPContext *base_ctx_;
    LocalTPContext *concrete_ctx_;
};

// =============================================================================
// Interface Metadata Tests
// =============================================================================

/**
 * @test Verify degree() through ITPContext* base pointer
 *
 * Tests that degree() returns the correct number of devices when accessed
 * through the base interface pointer.
 */
TEST_F(Test__ITPContext_LocalTP, Degree_ThroughBaseInterface)
{
    EXPECT_EQ(base_ctx_->degree(), 2);
}

/**
 * @test Verify isLocal() returns true for LocalTPContext
 *
 * LocalTPContext represents LOCAL tensor parallelism (intra-rank), so
 * isLocal() must always return true.
 */
TEST_F(Test__ITPContext_LocalTP, IsLocal_ReturnsTrue)
{
    EXPECT_TRUE(base_ctx_->isLocal());
}

/**
 * @test Verify isGlobal() returns false for LocalTPContext
 *
 * isGlobal() is the logical inverse of isLocal(), so for LOCAL TP it
 * must return false.
 */
TEST_F(Test__ITPContext_LocalTP, IsGlobal_ReturnsFalse)
{
    EXPECT_FALSE(base_ctx_->isGlobal());
}

/**
 * @test Verify isLocal() and isGlobal() are exact inverses
 *
 * The ITPContext interface defines isGlobal() as !isLocal(), so these
 * methods must always be logical inverses.
 */
TEST_F(Test__ITPContext_LocalTP, IsLocal_IsGlobal_AreInverses)
{
    EXPECT_EQ(base_ctx_->isLocal(), !base_ctx_->isGlobal());
    EXPECT_EQ(base_ctx_->isGlobal(), !base_ctx_->isLocal());
}

// =============================================================================
// myIndex() and setCurrentDeviceIndex() Tests
// =============================================================================

/**
 * @test Verify myIndex() through ITPContext* after setCurrentDeviceIndex()
 *
 * Tests the complete workflow: create context, cast to base, set index,
 * read index through base pointer.
 */
TEST_F(Test__ITPContext_LocalTP, MyIndex_ThroughBasePointer)
{
    concrete_ctx_->setCurrentDeviceIndex(0);
    EXPECT_EQ(base_ctx_->myIndex(), 0);

    concrete_ctx_->setCurrentDeviceIndex(1);
    EXPECT_EQ(base_ctx_->myIndex(), 1);
}

/**
 * @test Verify myIndex() throws when not set
 *
 * In orchestrator-driven LOCAL TP, myIndex() requires setCurrentDeviceIndex()
 * to be called first. Calling myIndex() without setting should throw.
 */
TEST_F(Test__ITPContext_LocalTP, MyIndex_ThrowsWhenNotSet)
{
    // Create a fresh context without setting index
    auto fresh_ctx = createLocalTPContext(
        {cpu0_, cpu1_},
        {},
        CollectiveBackendType::HOST);

    ITPContext *base = fresh_ctx.get();

    // myIndex() should throw std::runtime_error when never set
    EXPECT_THROW(base->myIndex(), std::runtime_error);
}

/**
 * @test Verify setCurrentDeviceIndex() accepts valid indices
 *
 * For a 2-device context, valid indices are 0 and 1.
 */
TEST_F(Test__ITPContext_LocalTP, SetCurrentDeviceIndex_ValidIndices)
{
    EXPECT_NO_THROW(concrete_ctx_->setCurrentDeviceIndex(0));
    EXPECT_EQ(base_ctx_->myIndex(), 0);

    EXPECT_NO_THROW(concrete_ctx_->setCurrentDeviceIndex(1));
    EXPECT_EQ(base_ctx_->myIndex(), 1);
}

/**
 * @test Verify setCurrentDeviceIndex() throws for out-of-range indices
 *
 * For a 2-device context, index >= 2 should throw std::out_of_range.
 */
TEST_F(Test__ITPContext_LocalTP, SetCurrentDeviceIndex_OutOfRange_Throws)
{
    EXPECT_THROW(concrete_ctx_->setCurrentDeviceIndex(2), std::out_of_range);
    EXPECT_THROW(concrete_ctx_->setCurrentDeviceIndex(100), std::out_of_range);
    EXPECT_THROW(concrete_ctx_->setCurrentDeviceIndex(-1), std::out_of_range);
}

/**
 * @test Verify device index can be changed multiple times
 *
 * The orchestrator may switch between devices, so setCurrentDeviceIndex()
 * should allow changing the current device index repeatedly.
 */
TEST_F(Test__ITPContext_LocalTP, SetCurrentDeviceIndex_CanBeChanged)
{
    concrete_ctx_->setCurrentDeviceIndex(0);
    EXPECT_EQ(base_ctx_->myIndex(), 0);

    concrete_ctx_->setCurrentDeviceIndex(1);
    EXPECT_EQ(base_ctx_->myIndex(), 1);

    concrete_ctx_->setCurrentDeviceIndex(0);
    EXPECT_EQ(base_ctx_->myIndex(), 0);
}

// =============================================================================
// Polymorphic Usage Tests
// =============================================================================

/**
 * @test Verify LocalTPContext can be accessed through ITPContext*
 *
 * Tests that the interface works correctly when accessed through
 * a base class pointer, which is the pattern used in production code
 * (e.g., AllreduceStage holding an ITPContext*).
 */
TEST_F(Test__ITPContext_LocalTP, PolymorphicUsage_ThroughITPContextPointer)
{
    // Access through base interface pointer
    ITPContext *base_ptr = local_ctx_.get();
    ASSERT_NE(base_ptr, nullptr);

    // Verify degree() works
    EXPECT_EQ(base_ptr->degree(), 2);

    // Verify isLocal() works
    EXPECT_TRUE(base_ptr->isLocal());

    // Verify isGlobal() works
    EXPECT_FALSE(base_ptr->isGlobal());

    // Set device index and verify through base pointer
    concrete_ctx_->setCurrentDeviceIndex(1);
    EXPECT_EQ(base_ptr->myIndex(), 1);
}

/**
 * @test Verify LocalTPContext can be cast to ITPContext via dynamic_cast
 *
 * This validates that LocalTPContext properly implements the ITPContext
 * interface inheritance hierarchy.
 */
TEST_F(Test__ITPContext_LocalTP, DynamicCast_ToITPContext)
{
    // Should be able to dynamic_cast to ITPContext*
    ITPContext *dynamic_ptr = dynamic_cast<ITPContext *>(local_ctx_.get());
    ASSERT_NE(dynamic_ptr, nullptr);

    // The cast should preserve functionality
    EXPECT_EQ(dynamic_ptr->degree(), 2);
    EXPECT_TRUE(dynamic_ptr->isLocal());
}

// =============================================================================
// ILocalTPContext Interface Tests
// =============================================================================

/**
 * @test Verify devices() returns the configured devices
 *
 * Tests that the ILocalTPContext-specific method devices() returns
 * the correct device list.
 */
TEST_F(Test__ITPContext_LocalTP, Devices_ReturnsConfiguredDevices)
{
    const auto &devices = local_ctx_->devices();

    ASSERT_EQ(devices.size(), 2);
    EXPECT_EQ(devices[0], cpu0_);
    EXPECT_EQ(devices[1], cpu1_);
}

/**
 * @test Verify weights() returns equal weights for default configuration
 *
 * When created with empty weights vector, weights should default to
 * equal distribution.
 */
TEST_F(Test__ITPContext_LocalTP, Weights_DefaultsToEqual)
{
    const auto &weights = local_ctx_->weights();

    ASSERT_EQ(weights.size(), 2);
    EXPECT_FLOAT_EQ(weights[0], 0.5f);
    EXPECT_FLOAT_EQ(weights[1], 0.5f);
}

/**
 * @test Verify backend() returns the configured backend type
 */
TEST_F(Test__ITPContext_LocalTP, Backend_ReturnsConfiguredType)
{
    EXPECT_EQ(local_ctx_->backend(), CollectiveBackendType::HOST);
}

// =============================================================================
// Context with Custom Weights
// =============================================================================

class Test__ITPContext_LocalTP_CustomWeights : public ::testing::Test
{
protected:
    void SetUp() override
    {
        cpu0_ = GlobalDeviceAddress::cpu(0);
        cpu1_ = GlobalDeviceAddress::cpu(1);

        // Create context with custom weights (70/30 split)
        local_ctx_ = createLocalTPContext(
            {cpu0_, cpu1_},
            {0.7f, 0.3f},
            CollectiveBackendType::HOST);

        base_ctx_ = local_ctx_.get();
    }

    GlobalDeviceAddress cpu0_;
    GlobalDeviceAddress cpu1_;
    std::unique_ptr<ILocalTPContext> local_ctx_;
    ITPContext *base_ctx_;
};

/**
 * @test Verify custom weights are preserved
 */
TEST_F(Test__ITPContext_LocalTP_CustomWeights, Weights_PreservesCustomValues)
{
    const auto &weights = local_ctx_->weights();

    ASSERT_EQ(weights.size(), 2);
    EXPECT_FLOAT_EQ(weights[0], 0.7f);
    EXPECT_FLOAT_EQ(weights[1], 0.3f);
}

/**
 * @test Verify degree() is correct with custom weights
 */
TEST_F(Test__ITPContext_LocalTP_CustomWeights, Degree_WithCustomWeights)
{
    EXPECT_EQ(base_ctx_->degree(), 2);
}

// =============================================================================
// Multi-Device Context Tests
// =============================================================================

class Test__ITPContext_LocalTP_ThreeDevices : public ::testing::Test
{
protected:
    void SetUp() override
    {
        cpu0_ = GlobalDeviceAddress::cpu(0);
        cpu1_ = GlobalDeviceAddress::cpu(1);
        cpu2_ = GlobalDeviceAddress::cpu(2);

        local_ctx_ = createLocalTPContext(
            {cpu0_, cpu1_, cpu2_},
            {},  // Equal weights
            CollectiveBackendType::HOST);

        base_ctx_ = local_ctx_.get();
        concrete_ctx_ = dynamic_cast<LocalTPContext *>(local_ctx_.get());
    }

    GlobalDeviceAddress cpu0_;
    GlobalDeviceAddress cpu1_;
    GlobalDeviceAddress cpu2_;
    std::unique_ptr<ILocalTPContext> local_ctx_;
    ITPContext *base_ctx_;
    LocalTPContext *concrete_ctx_;
};

/**
 * @test Verify degree() for 3-device context
 */
TEST_F(Test__ITPContext_LocalTP_ThreeDevices, Degree_ReturnsThree)
{
    EXPECT_EQ(base_ctx_->degree(), 3);
}

/**
 * @test Verify all device indices are accessible
 */
TEST_F(Test__ITPContext_LocalTP_ThreeDevices, AllDeviceIndices_Accessible)
{
    for (int i = 0; i < 3; ++i)
    {
        EXPECT_NO_THROW(concrete_ctx_->setCurrentDeviceIndex(i));
        EXPECT_EQ(base_ctx_->myIndex(), i);
    }
}

/**
 * @test Verify equal weights for 3-device context
 */
TEST_F(Test__ITPContext_LocalTP_ThreeDevices, Weights_EqualForThreeDevices)
{
    const auto &weights = local_ctx_->weights();

    ASSERT_EQ(weights.size(), 3);

    // Each device should get 1/3 of the work
    float expected = 1.0f / 3.0f;
    EXPECT_NEAR(weights[0], expected, 1e-5f);
    EXPECT_NEAR(weights[1], expected, 1e-5f);
    EXPECT_NEAR(weights[2], expected, 1e-5f);
}
