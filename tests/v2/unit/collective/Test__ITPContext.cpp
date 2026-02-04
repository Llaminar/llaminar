/**
 * @file Test__ITPContext.cpp
 * @brief Unit tests for ITPContext base interface
 * @author David Sanftenberg
 * @date February 2026
 *
 * Tests for the ITPContext base interface:
 * - Interface compliance for LocalTPContext
 * - isLocal() / isGlobal() for LOCAL TP
 * - myIndex() and setCurrentDeviceIndex() functionality
 * - Polymorphic usage through ITPContext*
 */

#include <gtest/gtest.h>
#include "collective/ITPContext.h"
#include "collective/ILocalTPContext.h"
#include "collective/LocalTPContext.h"
#include "backends/GlobalDeviceAddress.h"
#include "config/OrchestrationConfig.h"

using namespace llaminar2;

// =============================================================================
// Test Fixture
// =============================================================================

class Test__ITPContext : public ::testing::Test
{
protected:
    void SetUp() override
    {
        cuda0_ = GlobalDeviceAddress::cuda(0, 0);
        cuda1_ = GlobalDeviceAddress::cuda(1, 0);
        cpu0_ = GlobalDeviceAddress::cpu(0);
    }

    /**
     * @brief Helper to create a LocalTPContext and get the concrete pointer
     *
     * createLocalTPContext returns unique_ptr<ILocalTPContext>, but some tests
     * need access to setCurrentDeviceIndex() which is on the concrete class.
     * This helper creates the context and stores it while returning a pointer
     * to the concrete LocalTPContext.
     */
    LocalTPContext *createConcreteContext(
        std::vector<GlobalDeviceAddress> devices,
        std::vector<float> weights = {})
    {
        // Create via factory and store in the vector
        contexts_.push_back(std::make_unique<LocalTPContext>(
            std::move(devices),
            std::move(weights),
            CollectiveBackendType::HOST));
        return contexts_.back().get();
    }

    GlobalDeviceAddress cuda0_;
    GlobalDeviceAddress cuda1_;
    GlobalDeviceAddress cpu0_;
    std::vector<std::unique_ptr<LocalTPContext>> contexts_;
};

// =============================================================================
// Interface Compliance Tests
// =============================================================================

/**
 * @test Verify LocalTPContext can be cast to ITPContext*
 *
 * This validates that LocalTPContext properly implements the ITPContext interface
 * and can be used polymorphically.
 */
TEST_F(Test__ITPContext, LocalTPContext_ImplementsITPContext)
{
    auto ctx = createLocalTPContext({cuda0_, cuda1_}, {}, CollectiveBackendType::HOST);

    // Should be able to cast to ITPContext*
    ITPContext *base_ptr = ctx.get();
    ASSERT_NE(base_ptr, nullptr);

    // Should also work with dynamic_cast
    ITPContext *dynamic_ptr = dynamic_cast<ITPContext *>(ctx.get());
    ASSERT_NE(dynamic_ptr, nullptr);
}

/**
 * @test Test isLocal() returns true for LocalTPContext
 *
 * LOCAL TP contexts are intra-rank (within a single MPI rank), so isLocal()
 * must return true.
 */
TEST_F(Test__ITPContext, LocalTPContext_IsLocal_ReturnsTrue)
{
    auto ctx = createLocalTPContext({cuda0_}, {}, CollectiveBackendType::HOST);

    EXPECT_TRUE(ctx->isLocal());
}

/**
 * @test Test isGlobal() returns false for LocalTPContext
 *
 * isGlobal() is the logical inverse of isLocal(). For LOCAL TP contexts,
 * it must return false.
 */
TEST_F(Test__ITPContext, LocalTPContext_IsGlobal_ReturnsFalse)
{
    auto ctx = createLocalTPContext({cuda0_}, {}, CollectiveBackendType::HOST);

    EXPECT_FALSE(ctx->isGlobal());
}

/**
 * @test Test degree() through ITPContext* matches device count
 *
 * Validates that degree() returns correct values when accessed through
 * the base interface pointer.
 */
TEST_F(Test__ITPContext, LocalTPContext_Degree_MatchesDeviceCount)
{
    // Single device
    {
        auto ctx = createLocalTPContext({cuda0_}, {}, CollectiveBackendType::HOST);
        ITPContext *base_ptr = ctx.get();
        EXPECT_EQ(base_ptr->degree(), 1);
    }

    // Two devices
    {
        auto ctx = createLocalTPContext({cuda0_, cuda1_}, {}, CollectiveBackendType::HOST);
        ITPContext *base_ptr = ctx.get();
        EXPECT_EQ(base_ptr->degree(), 2);
    }

    // Three devices (mixed)
    {
        auto ctx = createLocalTPContext({cuda0_, cuda1_, cpu0_}, {}, CollectiveBackendType::HOST);
        ITPContext *base_ptr = ctx.get();
        EXPECT_EQ(base_ptr->degree(), 3);
    }
}

// =============================================================================
// myIndex() and setCurrentDeviceIndex() Tests
// =============================================================================

/**
 * @test Test myIndex() throws when not set
 *
 * In orchestrator-driven LOCAL TP, myIndex() requires setCurrentDeviceIndex()
 * to be called first. Calling myIndex() without setting the index should throw.
 */
TEST_F(Test__ITPContext, MyIndex_ThrowsWhenNotSet)
{
    auto *ctx = createConcreteContext({cuda0_, cuda1_});

    // myIndex() should throw std::runtime_error when never set
    EXPECT_THROW(ctx->myIndex(), std::runtime_error);
}

/**
 * @test Test setting a valid index works
 *
 * setCurrentDeviceIndex() should accept valid indices without throwing.
 */
TEST_F(Test__ITPContext, SetCurrentDeviceIndex_ValidIndex)
{
    auto *ctx = createConcreteContext({cuda0_, cuda1_});

    // Should not throw for valid indices
    EXPECT_NO_THROW(ctx->setCurrentDeviceIndex(0));
    EXPECT_NO_THROW(ctx->setCurrentDeviceIndex(1));
}

/**
 * @test Test out-of-range index throws std::out_of_range
 *
 * setCurrentDeviceIndex() should throw when the index is >= degree().
 */
TEST_F(Test__ITPContext, SetCurrentDeviceIndex_OutOfRange_Throws)
{
    auto *ctx = createConcreteContext({cuda0_, cuda1_});

    // degree() is 2, so valid indices are 0 and 1
    EXPECT_THROW(ctx->setCurrentDeviceIndex(2), std::out_of_range);
    EXPECT_THROW(ctx->setCurrentDeviceIndex(3), std::out_of_range);
    EXPECT_THROW(ctx->setCurrentDeviceIndex(100), std::out_of_range);

    // Negative indices should also be invalid (if int is signed)
    EXPECT_THROW(ctx->setCurrentDeviceIndex(-1), std::out_of_range);
}

/**
 * @test Test myIndex() returns what was set
 *
 * After calling setCurrentDeviceIndex(), myIndex() should return the same value.
 */
TEST_F(Test__ITPContext, MyIndex_ReturnsSetValue)
{
    auto *ctx = createConcreteContext({cuda0_, cuda1_});

    ctx->setCurrentDeviceIndex(0);
    EXPECT_EQ(ctx->myIndex(), 0);

    ctx->setCurrentDeviceIndex(1);
    EXPECT_EQ(ctx->myIndex(), 1);
}

/**
 * @test Test setCurrentDeviceIndex() can be called multiple times
 *
 * The orchestrator may switch between devices, so setCurrentDeviceIndex()
 * should allow changing the current device index.
 */
TEST_F(Test__ITPContext, MyIndex_CanBeChanged)
{
    auto *ctx = createConcreteContext({cuda0_, cuda1_, cpu0_});

    // Set to device 0
    ctx->setCurrentDeviceIndex(0);
    EXPECT_EQ(ctx->myIndex(), 0);

    // Change to device 2
    ctx->setCurrentDeviceIndex(2);
    EXPECT_EQ(ctx->myIndex(), 2);

    // Change back to device 1
    ctx->setCurrentDeviceIndex(1);
    EXPECT_EQ(ctx->myIndex(), 1);

    // And back to device 0
    ctx->setCurrentDeviceIndex(0);
    EXPECT_EQ(ctx->myIndex(), 0);
}

// =============================================================================
// Polymorphic Usage Tests
// =============================================================================

/**
 * @test Test using LocalTPContext via ITPContext* for degree() and isLocal()
 *
 * This validates that the interface works correctly when accessed through
 * a base class pointer, which is the pattern used in production code
 * (e.g., AllreduceStage holding an ITPContext*).
 */
TEST_F(Test__ITPContext, PolymorphicUsage_ThroughITPContextPointer)
{
    auto *ctx = createConcreteContext({cuda0_, cuda1_});

    // Access through base interface pointer
    ITPContext *base_ptr = ctx;

    // Verify degree() works
    EXPECT_EQ(base_ptr->degree(), 2);

    // Verify isLocal() works (must use base interface method)
    EXPECT_TRUE(base_ptr->isLocal());

    // Verify isGlobal() works (defined in ITPContext, not virtual)
    EXPECT_FALSE(base_ptr->isGlobal());

    // Set device index and verify through base pointer
    ctx->setCurrentDeviceIndex(1);
    EXPECT_EQ(base_ptr->myIndex(), 1);
}

/**
 * @test Test that myIndex() works through ITPContext* after setCurrentDeviceIndex()
 *
 * Validates the complete workflow: create context, cast to base, set index,
 * read index through base pointer.
 */
TEST_F(Test__ITPContext, PolymorphicUsage_MyIndex_ThroughBasePointer)
{
    auto *ctx = createConcreteContext({cuda0_, cuda1_, cpu0_});

    // Cast to base interface
    ITPContext *base_ptr = ctx;

    // Before setting, myIndex() should throw
    EXPECT_THROW(base_ptr->myIndex(), std::runtime_error);

    // Set index via concrete class
    ctx->setCurrentDeviceIndex(2);

    // Read index through base pointer
    EXPECT_EQ(base_ptr->myIndex(), 2);
}

// =============================================================================
// Edge Case Tests
// =============================================================================

/**
 * @test Test single-device context myIndex behavior
 *
 * Even with a single device, myIndex() should throw when not set
 * and work correctly after setting to 0.
 */
TEST_F(Test__ITPContext, SingleDevice_MyIndex_Behavior)
{
    auto *ctx = createConcreteContext({cuda0_});

    // Should throw when not set
    EXPECT_THROW(ctx->myIndex(), std::runtime_error);

    // Only valid index is 0
    EXPECT_NO_THROW(ctx->setCurrentDeviceIndex(0));
    EXPECT_EQ(ctx->myIndex(), 0);

    // Index 1 is out of range
    EXPECT_THROW(ctx->setCurrentDeviceIndex(1), std::out_of_range);
}

/**
 * @test Test isLocal/isGlobal consistency
 *
 * isGlobal() is defined as !isLocal() in ITPContext. For all LocalTPContext
 * instances, these should be logical inverses.
 */
TEST_F(Test__ITPContext, IsLocal_IsGlobal_AreInverses)
{
    auto ctx = createLocalTPContext({cuda0_, cuda1_}, {}, CollectiveBackendType::HOST);

    // Test through base pointer
    ITPContext *base_ptr = ctx.get();

    // isLocal() and isGlobal() should be exact inverses
    EXPECT_EQ(base_ptr->isLocal(), !base_ptr->isGlobal());
    EXPECT_EQ(base_ptr->isGlobal(), !base_ptr->isLocal());

    // For LOCAL TP specifically
    EXPECT_TRUE(base_ptr->isLocal());
    EXPECT_FALSE(base_ptr->isGlobal());
}

