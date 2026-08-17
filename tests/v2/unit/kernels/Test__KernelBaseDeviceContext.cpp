/**
 * @file Test__KernelBaseDeviceContext.cpp
 * @brief Unit tests for CUDAKernelBase and ROCmKernelBase device context support
 *
 * Tests the Phase 4 GPU Device Context Refactor additions to kernel base classes:
 * - setDeviceContext() / deviceContext() / hasDeviceContext()
 * - getStream() / getBlasHandle() helpers
 *
 * These tests use mock contexts and don't require actual GPU hardware.
 *
 * @author David Sanftenberg
 * @date February 2026
 */

#include <gtest/gtest.h>
#include "kernels/cuda/CUDAKernelBase.h"
#include "kernels/rocm/ROCmKernelBase.h"
#include "backends/IWorkerGPUContext.h"
#include "backends/IGPUGraphCapture.h"

using namespace llaminar2;

// ============================================================================
// Mock GPU Context for Testing
// ============================================================================

/**
 * @brief Mock implementation of IWorkerGPUContext for unit testing
 *
 * Provides predictable return values for testing kernel base class integration.
 * Does not require actual GPU hardware.
 */
class MockGPUContext : public IWorkerGPUContext
{
public:
    explicit MockGPUContext(
        int device_ordinal,
        bool initialized = true,
        bool owns_current_thread = true)
        : device_ordinal_(device_ordinal),
          initialized_(initialized),
          owns_current_thread_(owns_current_thread)
    {
    }

    // Device Info
    int deviceOrdinal() const override { return device_ordinal_; }
    std::string deviceName() const override { return "MockGPU-" + std::to_string(device_ordinal_); }
    bool isInitialized() const override { return initialized_; }

    // Work Submission
    bool ownsCurrentThread() const noexcept override
    {
        return owns_current_thread_;
    }

    std::future<void> submitAsync(std::function<void()> work) override
    {
        ++queued_submission_count_;
        std::packaged_task<void()> task(std::move(work));
        auto future = task.get_future();
        task();
        return future;
    }

    // Stream Access - return mock pointers
    void *defaultStream() override { return mock_stream_; }
    void *createStream() override { return mock_stream_; }
    void destroyStream(void * /*stream*/) override {}
    void *getOrCreateAuxiliaryStream(const std::string & /*name*/, bool *created = nullptr) override
    {
        if (created)
            *created = false;
        return mock_auxiliary_stream_;
    }
    void resetAuxiliaryStreams() override {}

    // Event Access - return mock pointers
    void *createEvent() override { return mock_event_; }
    void destroyEvent(void * /*event*/) override {}
    void recordEvent(void * /*event*/, void * /*stream*/) override {}
    void waitEvent(void * /*event*/, void * /*stream*/) override {}
    void synchronizeEvent(void * /*event*/) override {}
    float eventElapsedTime(void * /*start*/, void * /*stop*/) override { return 0.0f; }

    // Library Handles - return mock pointers
    void *blasHandle() override { return mock_blas_handle_; }
    void *blasLtHandle() override { return mock_blas_lt_handle_; }

    // Collective Communicator
    void setCollectiveComm(void *comm) override { mock_comm_ = comm; }
    void *collectiveComm() const override { return mock_comm_; }

    // Synchronization
    void synchronize() override {}
    void synchronizeStream(void * /*stream*/) override {}
    bool insertStreamDependency(void * /*dependent_stream*/, void * /*dependency_stream*/) override { return true; }

    // Graph Capture
    std::unique_ptr<IGPUGraphCapture> createGraphCapture() override { return nullptr; }
    std::unique_ptr<IGPUGraphCapture> createGraphCapture(void * /*stream*/) override { return nullptr; }

    // Test helpers - allow setting mock values
    void setMockStream(void *stream) { mock_stream_ = stream; }
    void setMockBlasHandle(void *handle) { mock_blas_handle_ = handle; }
    void setMockBlasLtHandle(void *handle) { mock_blas_lt_handle_ = handle; }
    void setInitialized(bool init) { initialized_ = init; }
    int queuedSubmissionCount() const { return queued_submission_count_; }

private:
    int device_ordinal_;
    bool initialized_;
    bool owns_current_thread_ = true;
    int queued_submission_count_ = 0;
    void *mock_stream_ = reinterpret_cast<void *>(0xDEADBEEF);
    void *mock_auxiliary_stream_ = reinterpret_cast<void *>(0xDEADCAFE);
    void *mock_event_ = reinterpret_cast<void *>(0xCAFEBABE);
    void *mock_blas_handle_ = reinterpret_cast<void *>(0x12345678);
    void *mock_blas_lt_handle_ = reinterpret_cast<void *>(0x87654321);
    void *mock_comm_ = nullptr;
};

TEST(Test__WorkerGPUContextSubmission, NestedSynchronousWorkExecutesInline)
{
    MockGPUContext context(/*device_ordinal=*/0,
                           /*initialized=*/true,
                           /*owns_current_thread=*/true);
    bool ran = false;

    context.submitAndWait([&ran]() { ran = true; });

    EXPECT_TRUE(ran);
    EXPECT_EQ(context.queuedSubmissionCount(), 0);
}

TEST(Test__WorkerGPUContextSubmission, ForeignSynchronousWorkUsesQueueAndPropagatesFailure)
{
    MockGPUContext context(/*device_ordinal=*/0,
                           /*initialized=*/true,
                           /*owns_current_thread=*/false);

    EXPECT_THROW(
        context.submitAndWait(
            []() { throw std::runtime_error("worker setup failed"); }),
        std::runtime_error);
    EXPECT_EQ(context.queuedSubmissionCount(), 1);
}

// ============================================================================
// Test Derived Classes (Concrete implementations of base classes)
// ============================================================================

/**
 * @brief Concrete CUDA kernel for testing CUDAKernelBase
 */
class TestCUDAKernel : public CUDAKernelBase
{
public:
    TestCUDAKernel() = default;

    // Expose protected members for testing
    using CUDAKernelBase::device_ctx_;
    using CUDAKernelBase::workspace_;
};

/**
 * @brief Concrete ROCm kernel for testing ROCmKernelBase
 */
class TestROCmKernel : public ROCmKernelBase
{
public:
    TestROCmKernel() = default;

    // Expose protected members for testing
    using ROCmKernelBase::device_ctx_;
    using ROCmKernelBase::workspace_;
};

// ============================================================================
// CUDAKernelBase Device Context Tests
// ============================================================================

class Test__CUDAKernelBaseDeviceContext : public ::testing::Test
{
protected:
    void SetUp() override
    {
        kernel_ = std::make_unique<TestCUDAKernel>();
        mock_ctx_ = std::make_unique<MockGPUContext>(0);
    }

    std::unique_ptr<TestCUDAKernel> kernel_;
    std::unique_ptr<MockGPUContext> mock_ctx_;
};

TEST_F(Test__CUDAKernelBaseDeviceContext, InitiallyHasNoContext)
{
    EXPECT_FALSE(kernel_->hasDeviceContext());
    EXPECT_EQ(kernel_->deviceContext(), nullptr);
}

TEST_F(Test__CUDAKernelBaseDeviceContext, SetDeviceContext_StoresContext)
{
    kernel_->setDeviceContext(mock_ctx_.get());

    EXPECT_TRUE(kernel_->hasDeviceContext());
    EXPECT_EQ(kernel_->deviceContext(), mock_ctx_.get());
}

TEST_F(Test__CUDAKernelBaseDeviceContext, SetDeviceContext_CanClear)
{
    kernel_->setDeviceContext(mock_ctx_.get());
    EXPECT_TRUE(kernel_->hasDeviceContext());

    kernel_->setDeviceContext(nullptr);
    EXPECT_FALSE(kernel_->hasDeviceContext());
    EXPECT_EQ(kernel_->deviceContext(), nullptr);
}

TEST_F(Test__CUDAKernelBaseDeviceContext, GetStreamRejectsMissingOwnershipBeforeLaunch)
{
    EXPECT_THROW(kernel_->getStream(), std::runtime_error);
}

TEST_F(Test__CUDAKernelBaseDeviceContext, ContextStreamDoesNotBecomeAnImplicitKernelBinding)
{
    void *expected_stream = reinterpret_cast<void *>(0xABCDEF);
    mock_ctx_->setMockStream(expected_stream);
    kernel_->setDeviceContext(mock_ctx_.get());

    EXPECT_FALSE(kernel_->hasExplicitGPUStream());
    EXPECT_THROW(kernel_->getStream(), std::runtime_error);

    kernel_->bindGPUStream(ExplicitGPUStream{expected_stream});
    EXPECT_TRUE(kernel_->hasExplicitGPUStream());
    EXPECT_EQ(kernel_->getStream(), expected_stream);

    kernel_->clearGPUStreamBinding();
    EXPECT_FALSE(kernel_->hasExplicitGPUStream());
    EXPECT_THROW(kernel_->getStream(), std::runtime_error);
}

TEST_F(Test__CUDAKernelBaseDeviceContext, GetBlasHandle_ReturnsNullWithoutContext)
{
    EXPECT_EQ(kernel_->getBlasHandle(), nullptr);
}

TEST_F(Test__CUDAKernelBaseDeviceContext, GetBlasHandle_ReturnsContextHandle)
{
    void *expected_handle = reinterpret_cast<void *>(0x987654);
    mock_ctx_->setMockBlasHandle(expected_handle);
    kernel_->setDeviceContext(mock_ctx_.get());

    EXPECT_EQ(kernel_->getBlasHandle(), expected_handle);
}

TEST_F(Test__CUDAKernelBaseDeviceContext, WorkspaceAndContextAreIndependent)
{
    // Verify workspace and device context are separate features
    kernel_->setDeviceContext(mock_ctx_.get());

    EXPECT_TRUE(kernel_->hasDeviceContext());
    EXPECT_FALSE(kernel_->hasWorkspace()); // Workspace not bound

    EXPECT_EQ(kernel_->getWorkspace(), nullptr);
    EXPECT_NE(kernel_->deviceContext(), nullptr);
}

// ============================================================================
// ROCmKernelBase Device Context Tests
// ============================================================================

class Test__ROCmKernelBaseDeviceContext : public ::testing::Test
{
protected:
    void SetUp() override
    {
        kernel_ = std::make_unique<TestROCmKernel>();
        mock_ctx_ = std::make_unique<MockGPUContext>(1);
    }

    std::unique_ptr<TestROCmKernel> kernel_;
    std::unique_ptr<MockGPUContext> mock_ctx_;
};

TEST_F(Test__ROCmKernelBaseDeviceContext, InitiallyHasNoContext)
{
    EXPECT_FALSE(kernel_->hasDeviceContext());
    EXPECT_EQ(kernel_->deviceContext(), nullptr);
}

TEST_F(Test__ROCmKernelBaseDeviceContext, SetDeviceContext_StoresContext)
{
    kernel_->setDeviceContext(mock_ctx_.get());

    EXPECT_TRUE(kernel_->hasDeviceContext());
    EXPECT_EQ(kernel_->deviceContext(), mock_ctx_.get());
}

TEST_F(Test__ROCmKernelBaseDeviceContext, SetDeviceContext_CanClear)
{
    kernel_->setDeviceContext(mock_ctx_.get());
    EXPECT_TRUE(kernel_->hasDeviceContext());

    kernel_->setDeviceContext(nullptr);
    EXPECT_FALSE(kernel_->hasDeviceContext());
    EXPECT_EQ(kernel_->deviceContext(), nullptr);
}

TEST_F(Test__ROCmKernelBaseDeviceContext, GetStreamRejectsMissingOwnershipBeforeLaunch)
{
    EXPECT_THROW(kernel_->getStream(), std::runtime_error);
}

TEST_F(Test__ROCmKernelBaseDeviceContext, ContextStreamDoesNotBecomeAnImplicitKernelBinding)
{
    void *expected_stream = reinterpret_cast<void *>(0xFEDCBA);
    mock_ctx_->setMockStream(expected_stream);
    kernel_->setDeviceContext(mock_ctx_.get());

    EXPECT_FALSE(kernel_->hasExplicitGPUStream());
    EXPECT_THROW(kernel_->getStream(), std::runtime_error);

    kernel_->bindGPUStream(ExplicitGPUStream{expected_stream});
    EXPECT_TRUE(kernel_->hasExplicitGPUStream());
    EXPECT_EQ(kernel_->getStream(), expected_stream);

    kernel_->clearGPUStreamBinding();
    EXPECT_FALSE(kernel_->hasExplicitGPUStream());
    EXPECT_THROW(kernel_->getStream(), std::runtime_error);
}

TEST_F(Test__ROCmKernelBaseDeviceContext, GetBlasHandle_ReturnsNullWithoutContext)
{
    EXPECT_EQ(kernel_->getBlasHandle(), nullptr);
}

TEST_F(Test__ROCmKernelBaseDeviceContext, GetBlasHandle_ReturnsContextHandle)
{
    void *expected_handle = reinterpret_cast<void *>(0x456789);
    mock_ctx_->setMockBlasHandle(expected_handle);
    kernel_->setDeviceContext(mock_ctx_.get());

    EXPECT_EQ(kernel_->getBlasHandle(), expected_handle);
}

TEST_F(Test__ROCmKernelBaseDeviceContext, WorkspaceAndContextAreIndependent)
{
    // Verify workspace and device context are separate features
    kernel_->setDeviceContext(mock_ctx_.get());

    EXPECT_TRUE(kernel_->hasDeviceContext());
    EXPECT_FALSE(kernel_->hasWorkspace()); // Workspace not bound

    EXPECT_EQ(kernel_->getWorkspace(), nullptr);
    EXPECT_NE(kernel_->deviceContext(), nullptr);
}

// ============================================================================
// Cross-Context Tests
// ============================================================================

TEST(Test__KernelBaseDeviceContext, ExplicitGPUStreamRejectsNullBeforeKernelBinding)
{
    EXPECT_THROW((void)ExplicitGPUStream{nullptr}, std::invalid_argument);
}

TEST(Test__KernelBaseDeviceContext, MultipleKernelsCanShareContext)
{
    auto mock_ctx = std::make_unique<MockGPUContext>(0);
    void *shared_handle = reinterpret_cast<void *>(0xABCDE);
    mock_ctx->setMockBlasHandle(shared_handle);

    TestCUDAKernel kernel1;
    TestCUDAKernel kernel2;

    kernel1.setDeviceContext(mock_ctx.get());
    kernel2.setDeviceContext(mock_ctx.get());

    // Both kernels should see the same context and handle
    EXPECT_EQ(kernel1.deviceContext(), kernel2.deviceContext());
    EXPECT_EQ(kernel1.getBlasHandle(), kernel2.getBlasHandle());
    EXPECT_EQ(kernel1.getBlasHandle(), shared_handle);
}

TEST(Test__KernelBaseDeviceContext, KernelCanSwitchContexts)
{
    auto ctx1 = std::make_unique<MockGPUContext>(0);
    auto ctx2 = std::make_unique<MockGPUContext>(1);

    void *handle1 = reinterpret_cast<void *>(0x111);
    void *handle2 = reinterpret_cast<void *>(0x222);
    ctx1->setMockBlasHandle(handle1);
    ctx2->setMockBlasHandle(handle2);

    TestCUDAKernel kernel;

    // Start with context 1
    kernel.setDeviceContext(ctx1.get());
    EXPECT_EQ(kernel.getBlasHandle(), handle1);

    // Switch to context 2
    kernel.setDeviceContext(ctx2.get());
    EXPECT_EQ(kernel.getBlasHandle(), handle2);

    // Switch back to context 1
    kernel.setDeviceContext(ctx1.get());
    EXPECT_EQ(kernel.getBlasHandle(), handle1);
}
