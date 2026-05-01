/**
 * @file Test__GPUGraphCapture.cpp
 * @brief Unit tests for GPU Graph Capture/Replay infrastructure
 *
 * Tests:
 * - IGPUGraphCapture interface via concrete implementations
 * - HIPGraphCapture lifecycle (ROCm)
 * - CUDAGraphCapture lifecycle (CUDA)
 * - Factory method: IWorkerGPUContext::createGraphCapture()
 * - Error paths, move semantics, reset safety
 *
 * All GPU API calls are issued from the device context's worker thread
 * via submitAndWait() to maintain thread safety.
 *
 * @date January 2026
 */

#include <gtest/gtest.h>
#include "backends/GPUDeviceContextPool.h"
#include "backends/IWorkerGPUContext.h"
#include "backends/IGPUGraphCapture.h"

// NOTE: We cannot include both HIPGraphCapture.h and CUDAGraphCapture.h in the
// same translation unit because <hip/hip_runtime.h> and <cuda_runtime.h> have
// conflicting vector_types definitions. We verify concrete types via backendName()
// instead of dynamic_cast.

#include <memory>
#include <string>
#include <cstring>

using namespace llaminar2;

// ===========================================================================
// Skip macros for hardware availability
// ===========================================================================

#if defined(GPU_CONTEXT_TEST_BACKEND_CUDA)
#define SKIP_IF_NO_CUDA()                                      \
    do                                                         \
    {                                                          \
        ensureNvidiaFactoryRegistered();                       \
        if (!GPUDeviceContextPool::instance().hasNvidiaSupport()) \
            GTEST_SKIP() << "CUDA not available";              \
    } while (false)
#else
#define SKIP_IF_NO_CUDA() GTEST_SKIP() << "CUDA backend not linked in this test binary"
#endif

#if defined(GPU_CONTEXT_TEST_BACKEND_ROCM)
#define SKIP_IF_NO_ROCM()                                   \
    do                                                      \
    {                                                       \
        ensureAMDFactoryRegistered();                       \
        if (!GPUDeviceContextPool::instance().hasAMDSupport()) \
            GTEST_SKIP() << "ROCm not available";           \
    } while (false)
#else
#define SKIP_IF_NO_ROCM() GTEST_SKIP() << "ROCm backend not linked in this test binary"
#endif

#if defined(GPU_CONTEXT_TEST_BACKEND_ROCM)
#define SKIP_IF_NO_GPU() SKIP_IF_NO_ROCM()
#elif defined(GPU_CONTEXT_TEST_BACKEND_CUDA)
#define SKIP_IF_NO_GPU() SKIP_IF_NO_CUDA()
#else
#define SKIP_IF_NO_GPU() GTEST_SKIP() << "No GPU backend linked in this test binary"
#endif

// ===========================================================================
// 1. Interface / Enum Tests
// ===========================================================================

TEST(Test__GPUGraphCapture, GraphUpdateResultEnumValues)
{
    // Verify all enum values exist and are distinct
    auto success = GraphUpdateResult::Success;
    auto needs_reinstantiate = GraphUpdateResult::NeedsReinstantiate;
    auto failed = GraphUpdateResult::Failed;

    EXPECT_NE(success, needs_reinstantiate);
    EXPECT_NE(success, failed);
    EXPECT_NE(needs_reinstantiate, failed);
}

TEST(Test__GPUGraphCapture, BackendNameCheck_ROCm)
{
    SKIP_IF_NO_ROCM();
    auto &pool = GPUDeviceContextPool::instance();
    auto &ctx = pool.getAMDContext(0);

    ctx.submitAndWait([&]
                      {
        auto capture = ctx.createGraphCapture();
        ASSERT_NE(capture, nullptr);
        EXPECT_STREQ(capture->backendName(), "HIP"); });
}

TEST(Test__GPUGraphCapture, BackendNameCheck_CUDA)
{
    SKIP_IF_NO_CUDA();
    auto &pool = GPUDeviceContextPool::instance();
    auto &ctx = pool.getNvidiaContext(0);

    ctx.submitAndWait([&]
                      {
        auto capture = ctx.createGraphCapture();
        ASSERT_NE(capture, nullptr);
        EXPECT_STREQ(capture->backendName(), "CUDA"); });
}

// ===========================================================================
// 2. Factory Tests
// ===========================================================================

TEST(Test__GPUGraphCapture, CreateGraphCapture_ROCm)
{
    SKIP_IF_NO_ROCM();
    auto &pool = GPUDeviceContextPool::instance();
    auto &ctx = pool.getAMDContext(0);

    ctx.submitAndWait([&]
                      {
        auto capture = ctx.createGraphCapture();
        ASSERT_NE(capture, nullptr);

        // Verify concrete type via backendName() (no header inclusion needed)
        EXPECT_STREQ(capture->backendName(), "HIP") << "ROCm factory should produce HIP backend"; });
}

TEST(Test__GPUGraphCapture, CreateGraphCapture_CUDA)
{
    SKIP_IF_NO_CUDA();
    auto &pool = GPUDeviceContextPool::instance();
    auto &ctx = pool.getNvidiaContext(0);

    ctx.submitAndWait([&]
                      {
        auto capture = ctx.createGraphCapture();
        ASSERT_NE(capture, nullptr);

        // Verify concrete type via backendName() (no header inclusion needed)
        EXPECT_STREQ(capture->backendName(), "CUDA") << "CUDA factory should produce CUDA backend"; });
}

TEST(Test__GPUGraphCapture, CreateViaWorkerThread_ROCm)
{
    SKIP_IF_NO_ROCM();
    auto &pool = GPUDeviceContextPool::instance();
    auto &ctx = pool.getAMDContext(0);

    // Create within submitAndWait to ensure worker-thread safety
    std::unique_ptr<IGPUGraphCapture> capture_out;
    ctx.submitAndWait([&]
                      {
        capture_out = ctx.createGraphCapture();
        ASSERT_NE(capture_out, nullptr);
        EXPECT_STREQ(capture_out->backendName(), "HIP"); });

    // After submitAndWait, the capture object is valid (owned by us)
    EXPECT_NE(capture_out, nullptr);
}

TEST(Test__GPUGraphCapture, CreateViaWorkerThread_CUDA)
{
    SKIP_IF_NO_CUDA();
    auto &pool = GPUDeviceContextPool::instance();
    auto &ctx = pool.getNvidiaContext(0);

    // Create within submitAndWait to ensure worker-thread safety
    std::unique_ptr<IGPUGraphCapture> capture_out;
    ctx.submitAndWait([&]
                      {
        capture_out = ctx.createGraphCapture();
        ASSERT_NE(capture_out, nullptr);
        EXPECT_STREQ(capture_out->backendName(), "CUDA"); });

    // After submitAndWait, the capture object is valid (owned by us)
    EXPECT_NE(capture_out, nullptr);
}

// ===========================================================================
// 3. Initial State Tests
// ===========================================================================

TEST(Test__GPUGraphCapture, InitialState_NoExecutable_ROCm)
{
    SKIP_IF_NO_ROCM();
    auto &pool = GPUDeviceContextPool::instance();
    auto &ctx = pool.getAMDContext(0);

    ctx.submitAndWait([&]
                      {
        auto capture = ctx.createGraphCapture();
        ASSERT_NE(capture, nullptr);

        EXPECT_FALSE(capture->hasExecutable()) << "Fresh capture should have no executable";
        EXPECT_EQ(capture->nodeCount(), 0u) << "Fresh capture should have 0 nodes"; });
}

TEST(Test__GPUGraphCapture, InitialState_NoExecutable_CUDA)
{
    SKIP_IF_NO_CUDA();
    auto &pool = GPUDeviceContextPool::instance();
    auto &ctx = pool.getNvidiaContext(0);

    ctx.submitAndWait([&]
                      {
        auto capture = ctx.createGraphCapture();
        ASSERT_NE(capture, nullptr);

        EXPECT_FALSE(capture->hasExecutable()) << "Fresh capture should have no executable";
        EXPECT_EQ(capture->nodeCount(), 0u) << "Fresh capture should have 0 nodes"; });
}

// ===========================================================================
// 4. Lifecycle Tests
// ===========================================================================

TEST(Test__GPUGraphCapture, EmptyCaptureInstantiate_ROCm)
{
    SKIP_IF_NO_ROCM();
    auto &pool = GPUDeviceContextPool::instance();
    auto &ctx = pool.getAMDContext(0);

    ctx.submitAndWait([&]
                      {
        auto capture = ctx.createGraphCapture();
        ASSERT_NE(capture, nullptr);

        // Empty capture: beginCapture → endCapture → instantiate → launch
        EXPECT_TRUE(capture->beginCapture());
        EXPECT_TRUE(capture->endCapture());
        EXPECT_TRUE(capture->instantiate());
        EXPECT_TRUE(capture->hasExecutable());
        EXPECT_TRUE(capture->launch()); });
}

TEST(Test__GPUGraphCapture, EmptyCaptureInstantiate_CUDA)
{
    SKIP_IF_NO_CUDA();
    auto &pool = GPUDeviceContextPool::instance();
    auto &ctx = pool.getNvidiaContext(0);

    ctx.submitAndWait([&]
                      {
        auto capture = ctx.createGraphCapture();
        ASSERT_NE(capture, nullptr);

        // Empty capture: beginCapture → endCapture → instantiate → launch
        EXPECT_TRUE(capture->beginCapture());
        EXPECT_TRUE(capture->endCapture());
        EXPECT_TRUE(capture->instantiate());
        EXPECT_TRUE(capture->hasExecutable());
        EXPECT_TRUE(capture->launch()); });
}

TEST(Test__GPUGraphCapture, ResetClearsState_ROCm)
{
    SKIP_IF_NO_ROCM();
    auto &pool = GPUDeviceContextPool::instance();
    auto &ctx = pool.getAMDContext(0);

    ctx.submitAndWait([&]
                      {
        auto capture = ctx.createGraphCapture();
        ASSERT_NE(capture, nullptr);

        // First: capture and instantiate
        ASSERT_TRUE(capture->beginCapture());
        ASSERT_TRUE(capture->endCapture());
        ASSERT_TRUE(capture->instantiate());
        ASSERT_TRUE(capture->hasExecutable());

        // Now reset
        capture->reset();
        EXPECT_FALSE(capture->hasExecutable()) << "reset() should clear executable";
        EXPECT_EQ(capture->nodeCount(), 0u) << "reset() should clear node count"; });
}

TEST(Test__GPUGraphCapture, ResetClearsState_CUDA)
{
    SKIP_IF_NO_CUDA();
    auto &pool = GPUDeviceContextPool::instance();
    auto &ctx = pool.getNvidiaContext(0);

    ctx.submitAndWait([&]
                      {
        auto capture = ctx.createGraphCapture();
        ASSERT_NE(capture, nullptr);

        // First: capture and instantiate
        ASSERT_TRUE(capture->beginCapture());
        ASSERT_TRUE(capture->endCapture());
        ASSERT_TRUE(capture->instantiate());
        ASSERT_TRUE(capture->hasExecutable());

        // Now reset
        capture->reset();
        EXPECT_FALSE(capture->hasExecutable()) << "reset() should clear executable";
        EXPECT_EQ(capture->nodeCount(), 0u) << "reset() should clear node count"; });
}

TEST(Test__GPUGraphCapture, DoubleResetIsSafe_ROCm)
{
    SKIP_IF_NO_ROCM();
    auto &pool = GPUDeviceContextPool::instance();
    auto &ctx = pool.getAMDContext(0);

    ctx.submitAndWait([&]
                      {
        auto capture = ctx.createGraphCapture();
        ASSERT_NE(capture, nullptr);

        // Capture and instantiate first
        ASSERT_TRUE(capture->beginCapture());
        ASSERT_TRUE(capture->endCapture());
        ASSERT_TRUE(capture->instantiate());

        // Double reset should not crash
        capture->reset();
        capture->reset();

        EXPECT_FALSE(capture->hasExecutable());
        EXPECT_EQ(capture->nodeCount(), 0u); });
}

TEST(Test__GPUGraphCapture, DoubleResetIsSafe_CUDA)
{
    SKIP_IF_NO_CUDA();
    auto &pool = GPUDeviceContextPool::instance();
    auto &ctx = pool.getNvidiaContext(0);

    ctx.submitAndWait([&]
                      {
        auto capture = ctx.createGraphCapture();
        ASSERT_NE(capture, nullptr);

        // Capture and instantiate first
        ASSERT_TRUE(capture->beginCapture());
        ASSERT_TRUE(capture->endCapture());
        ASSERT_TRUE(capture->instantiate());

        // Double reset should not crash
        capture->reset();
        capture->reset();

        EXPECT_FALSE(capture->hasExecutable());
        EXPECT_EQ(capture->nodeCount(), 0u); });
}

// ===========================================================================
// 5. Error Path Tests
// ===========================================================================

TEST(Test__GPUGraphCapture, LaunchWithoutInstantiate_ROCm)
{
    SKIP_IF_NO_ROCM();
    auto &pool = GPUDeviceContextPool::instance();
    auto &ctx = pool.getAMDContext(0);

    ctx.submitAndWait([&]
                      {
        auto capture = ctx.createGraphCapture();
        ASSERT_NE(capture, nullptr);

        // launch() without instantiate should return false
        EXPECT_FALSE(capture->launch()) << "launch() without instantiate should fail"; });
}

TEST(Test__GPUGraphCapture, LaunchWithoutInstantiate_CUDA)
{
    SKIP_IF_NO_CUDA();
    auto &pool = GPUDeviceContextPool::instance();
    auto &ctx = pool.getNvidiaContext(0);

    ctx.submitAndWait([&]
                      {
        auto capture = ctx.createGraphCapture();
        ASSERT_NE(capture, nullptr);

        // launch() without instantiate should return false
        EXPECT_FALSE(capture->launch()) << "launch() without instantiate should fail"; });
}

TEST(Test__GPUGraphCapture, InstantiateWithoutCapture_ROCm)
{
    SKIP_IF_NO_ROCM();
    auto &pool = GPUDeviceContextPool::instance();
    auto &ctx = pool.getAMDContext(0);

    ctx.submitAndWait([&]
                      {
        auto capture = ctx.createGraphCapture();
        ASSERT_NE(capture, nullptr);

        // instantiate() without endCapture should return false
        EXPECT_FALSE(capture->instantiate()) << "instantiate() without capture should fail"; });
}

TEST(Test__GPUGraphCapture, InstantiateWithoutCapture_CUDA)
{
    SKIP_IF_NO_CUDA();
    auto &pool = GPUDeviceContextPool::instance();
    auto &ctx = pool.getNvidiaContext(0);

    ctx.submitAndWait([&]
                      {
        auto capture = ctx.createGraphCapture();
        ASSERT_NE(capture, nullptr);

        // instantiate() without endCapture should return false
        EXPECT_FALSE(capture->instantiate()) << "instantiate() without capture should fail"; });
}

TEST(Test__GPUGraphCapture, TryUpdateWithoutExecutable_ROCm)
{
    SKIP_IF_NO_ROCM();
    auto &pool = GPUDeviceContextPool::instance();
    auto &ctx = pool.getAMDContext(0);

    ctx.submitAndWait([&]
                      {
        auto capture = ctx.createGraphCapture();
        ASSERT_NE(capture, nullptr);

        // tryUpdate() without executable should return Failed
        auto result = capture->tryUpdate();
        EXPECT_EQ(result, GraphUpdateResult::Failed)
            << "tryUpdate() without executable should return Failed"; });
}

TEST(Test__GPUGraphCapture, TryUpdateWithoutExecutable_CUDA)
{
    SKIP_IF_NO_CUDA();
    auto &pool = GPUDeviceContextPool::instance();
    auto &ctx = pool.getNvidiaContext(0);

    ctx.submitAndWait([&]
                      {
        auto capture = ctx.createGraphCapture();
        ASSERT_NE(capture, nullptr);

        // tryUpdate() without executable should return Failed
        auto result = capture->tryUpdate();
        EXPECT_EQ(result, GraphUpdateResult::Failed)
            << "tryUpdate() without executable should return Failed"; });
}

// ===========================================================================
// 6. Move Semantics Tests
// ===========================================================================

// Move semantics are tested at the interface level via unique_ptr transfer.
// We cannot include concrete HIPGraphCapture.h / CUDAGraphCapture.h headers
// simultaneously (HIP/CUDA header collision), so we verify ownership transfer
// via the IGPUGraphCapture interface.

TEST(Test__GPUGraphCapture, OwnershipTransfer_ROCm)
{
    SKIP_IF_NO_ROCM();
    auto &pool = GPUDeviceContextPool::instance();
    auto &ctx = pool.getAMDContext(0);

    ctx.submitAndWait([&]
                      {
        auto capture = ctx.createGraphCapture();
        ASSERT_NE(capture, nullptr);

        // Capture and instantiate to get an executable
        ASSERT_TRUE(capture->beginCapture());
        ASSERT_TRUE(capture->endCapture());
        ASSERT_TRUE(capture->instantiate());
        ASSERT_TRUE(capture->hasExecutable());

        // Transfer ownership via unique_ptr move
        std::unique_ptr<IGPUGraphCapture> dest = std::move(capture);
        EXPECT_EQ(capture, nullptr);
        ASSERT_NE(dest, nullptr);

        // Destination should have the executable
        EXPECT_TRUE(dest->hasExecutable())
            << "Moved-to unique_ptr should own the executable";

        // Verify launch still works on moved-to object
        EXPECT_TRUE(dest->launch());
        EXPECT_STREQ(dest->backendName(), "HIP"); });
}

TEST(Test__GPUGraphCapture, OwnershipTransfer_CUDA)
{
    SKIP_IF_NO_CUDA();
    auto &pool = GPUDeviceContextPool::instance();
    auto &ctx = pool.getNvidiaContext(0);

    ctx.submitAndWait([&]
                      {
        auto capture = ctx.createGraphCapture();
        ASSERT_NE(capture, nullptr);

        // Capture and instantiate to get an executable
        ASSERT_TRUE(capture->beginCapture());
        ASSERT_TRUE(capture->endCapture());
        ASSERT_TRUE(capture->instantiate());
        ASSERT_TRUE(capture->hasExecutable());

        // Transfer ownership via unique_ptr move
        std::unique_ptr<IGPUGraphCapture> dest = std::move(capture);
        EXPECT_EQ(capture, nullptr);
        ASSERT_NE(dest, nullptr);

        // Destination should have the executable
        EXPECT_TRUE(dest->hasExecutable())
            << "Moved-to unique_ptr should own the executable";

        // Verify launch still works on moved-to object
        EXPECT_TRUE(dest->launch());
        EXPECT_STREQ(dest->backendName(), "CUDA"); });
}
