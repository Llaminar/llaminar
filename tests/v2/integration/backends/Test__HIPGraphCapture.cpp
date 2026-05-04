/**
 * @file Test__HIPGraphCapture.cpp
 * @brief Integration tests for HIP Graph Capture/Replay on ROCm GPUs
 *
 * Tests the IGPUGraphCapture interface via the HIPGraphCapture implementation:
 * - Factory creation via IWorkerGPUContext::createGraphCapture()
 * - Capture lifecycle: beginCapture → endCapture → instantiate → launch
 * - Reset/clear state
 * - Error paths (launch without instantiate, instantiate without capture)
 * - tryUpdate() behavior
 * - Ownership transfer via unique_ptr
 *
 * All GPU API calls are issued from the device context's worker thread
 * via submitAndWait() to maintain thread safety.
 */

#include <gtest/gtest.h>
#include "backends/GPUDeviceContextPool.h"
#include "backends/IWorkerGPUContext.h"
#include "backends/IGPUGraphCapture.h"

#include <memory>
#include <string>

using namespace llaminar2;

// ===========================================================================
// Test Fixture
// ===========================================================================

class Test__HIPGraphCapture : public ::testing::Test
{
protected:
    void SetUp() override
    {
        ensureAMDFactoryRegistered();
        if (!GPUDeviceContextPool::instance().hasAMDSupport())
            GTEST_SKIP() << "ROCm not available";
    }

    IWorkerGPUContext &ctx()
    {
        return GPUDeviceContextPool::instance().getAMDContext(0);
    }
};

// ===========================================================================
// Factory Tests
// ===========================================================================

TEST_F(Test__HIPGraphCapture, BackendNameIsHIP)
{
    ctx().submitAndWait([&] {
        auto capture = ctx().createGraphCapture();
        ASSERT_NE(capture, nullptr);
        EXPECT_STREQ(capture->backendName(), "HIP");
    });
}

TEST_F(Test__HIPGraphCapture, CreateReturnsNonNull)
{
    ctx().submitAndWait([&] {
        auto capture = ctx().createGraphCapture();
        ASSERT_NE(capture, nullptr);
    });
}

TEST_F(Test__HIPGraphCapture, CreateViaWorkerThread)
{
    std::unique_ptr<IGPUGraphCapture> capture_out;
    ctx().submitAndWait([&] {
        capture_out = ctx().createGraphCapture();
        ASSERT_NE(capture_out, nullptr);
        EXPECT_STREQ(capture_out->backendName(), "HIP");
    });
    EXPECT_NE(capture_out, nullptr);
}

// ===========================================================================
// Initial State Tests
// ===========================================================================

TEST_F(Test__HIPGraphCapture, InitialState_NoExecutable)
{
    ctx().submitAndWait([&] {
        auto capture = ctx().createGraphCapture();
        ASSERT_NE(capture, nullptr);
        EXPECT_FALSE(capture->hasExecutable()) << "Fresh capture should have no executable";
        EXPECT_EQ(capture->nodeCount(), 0u) << "Fresh capture should have 0 nodes";
    });
}

// ===========================================================================
// Lifecycle Tests
// ===========================================================================

TEST_F(Test__HIPGraphCapture, EmptyCaptureInstantiate)
{
    ctx().submitAndWait([&] {
        auto capture = ctx().createGraphCapture();
        ASSERT_NE(capture, nullptr);

        EXPECT_TRUE(capture->beginCapture());
        EXPECT_TRUE(capture->endCapture());
        EXPECT_TRUE(capture->instantiate());
        EXPECT_TRUE(capture->hasExecutable());
        EXPECT_TRUE(capture->launch());
    });
}

TEST_F(Test__HIPGraphCapture, ResetClearsState)
{
    ctx().submitAndWait([&] {
        auto capture = ctx().createGraphCapture();
        ASSERT_NE(capture, nullptr);

        ASSERT_TRUE(capture->beginCapture());
        ASSERT_TRUE(capture->endCapture());
        ASSERT_TRUE(capture->instantiate());
        ASSERT_TRUE(capture->hasExecutable());

        capture->reset();
        EXPECT_FALSE(capture->hasExecutable()) << "reset() should clear executable";
        EXPECT_EQ(capture->nodeCount(), 0u) << "reset() should clear node count";
    });
}

TEST_F(Test__HIPGraphCapture, DoubleResetIsSafe)
{
    ctx().submitAndWait([&] {
        auto capture = ctx().createGraphCapture();
        ASSERT_NE(capture, nullptr);

        ASSERT_TRUE(capture->beginCapture());
        ASSERT_TRUE(capture->endCapture());
        ASSERT_TRUE(capture->instantiate());

        capture->reset();
        capture->reset();

        EXPECT_FALSE(capture->hasExecutable());
        EXPECT_EQ(capture->nodeCount(), 0u);
    });
}

// ===========================================================================
// Error Path Tests
// ===========================================================================

TEST_F(Test__HIPGraphCapture, LaunchWithoutInstantiateFails)
{
    ctx().submitAndWait([&] {
        auto capture = ctx().createGraphCapture();
        ASSERT_NE(capture, nullptr);
        EXPECT_FALSE(capture->launch()) << "launch() without instantiate should fail";
    });
}

TEST_F(Test__HIPGraphCapture, InstantiateWithoutCaptureFails)
{
    ctx().submitAndWait([&] {
        auto capture = ctx().createGraphCapture();
        ASSERT_NE(capture, nullptr);
        EXPECT_FALSE(capture->instantiate()) << "instantiate() without capture should fail";
    });
}

TEST_F(Test__HIPGraphCapture, TryUpdateWithoutExecutableFails)
{
    ctx().submitAndWait([&] {
        auto capture = ctx().createGraphCapture();
        ASSERT_NE(capture, nullptr);

        auto result = capture->tryUpdate();
        EXPECT_EQ(result, GraphUpdateResult::Failed)
            << "tryUpdate() without executable should return Failed";
    });
}

// ===========================================================================
// Ownership Transfer Tests
// ===========================================================================

TEST_F(Test__HIPGraphCapture, OwnershipTransferViaMove)
{
    ctx().submitAndWait([&] {
        auto capture = ctx().createGraphCapture();
        ASSERT_NE(capture, nullptr);

        ASSERT_TRUE(capture->beginCapture());
        ASSERT_TRUE(capture->endCapture());
        ASSERT_TRUE(capture->instantiate());
        ASSERT_TRUE(capture->hasExecutable());

        std::unique_ptr<IGPUGraphCapture> dest = std::move(capture);
        EXPECT_EQ(capture, nullptr);
        ASSERT_NE(dest, nullptr);

        EXPECT_TRUE(dest->hasExecutable())
            << "Moved-to unique_ptr should own the executable";
        EXPECT_TRUE(dest->launch());
        EXPECT_STREQ(dest->backendName(), "HIP");
    });
}
