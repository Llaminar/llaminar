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
    explicit MockGPUContext(int device_ordinal, bool initialized = true)
        : device_ordinal_(device_ordinal), initialized_(initialized)
    {
    }

    // Device Info
    int deviceOrdinal() const override { return device_ordinal_; }
    std::string deviceName() const override { return "MockGPU-" + std::to_string(device_ordinal_); }
    bool isInitialized() const override { return initialized_; }

    // Work Submission (no-op for mock)
    void submitAndWait(std::function<void()> work) override { work(); }
    std::future<void> submitAsync(std::function<void()> work) override
    {
        work();
        std::promise<void> p;
        p.set_value();
        return p.get_future();
    }

    // Stream Access - return mock pointers
    void *defaultStream() override { return mock_stream_; }
    void *createStream() override { return mock_stream_; }
    void destroyStream(void * /*stream*/) override {}

    // Event Access - return mock pointers
    void *createEvent() override { return mock_event_; }
    void destroyEvent(void * /*event*/) override {}
    void recordEvent(void * /*event*/, void * /*stream*/) override {}
    void waitEvent(void * /*event*/, void * /*stream*/) override {}
    void synchronizeEvent(void * /*event*/) override {}

    // Library Handles - return mock pointers
    void *blasHandle() override { return mock_blas_handle_; }
    void *blasLtHandle() override { return mock_blas_lt_handle_; }

    // Collective Communicator
    void setCollectiveComm(void *comm) override { mock_comm_ = comm; }
    void *collectiveComm() const override { return mock_comm_; }

    // Synchronization
    void synchronize() override {}
    void synchronizeStream(void * /*stream*/) override {}
    void insertStreamDependency(void * /*dependent_stream*/, void * /*dependency_stream*/) override {}

    // Graph Capture
    std::unique_ptr<IGPUGraphCapture> createGraphCapture() override { return nullptr; }
    std::unique_ptr<IGPUGraphCapture> createGraphCapture(void * /*stream*/) override { return nullptr; }

    // Test helpers - allow setting mock values
    void setMockStream(void *stream) { mock_stream_ = stream; }
    void setMockBlasHandle(void *handle) { mock_blas_handle_ = handle; }
    void setMockBlasLtHandle(void *handle) { mock_blas_lt_handle_ = handle; }
    void setInitialized(bool init) { initialized_ = init; }

private:
    int device_ordinal_;
    bool initialized_;
    void *mock_stream_ = reinterpret_cast<void *>(0xDEADBEEF);
    void *mock_event_ = reinterpret_cast<void *>(0xCAFEBABE);
    void *mock_blas_handle_ = reinterpret_cast<void *>(0x12345678);
    void *mock_blas_lt_handle_ = reinterpret_cast<void *>(0x87654321);
    void *mock_comm_ = nullptr;
};

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

TEST_F(Test__CUDAKernelBaseDeviceContext, GetStream_ReturnsNullWithoutContext)
{
    EXPECT_EQ(kernel_->getStream(), nullptr);
}

TEST_F(Test__CUDAKernelBaseDeviceContext, GetStream_ReturnsContextStream)
{
    void *expected_stream = reinterpret_cast<void *>(0xABCDEF);
    mock_ctx_->setMockStream(expected_stream);
    kernel_->setDeviceContext(mock_ctx_.get());

    EXPECT_EQ(kernel_->getStream(), expected_stream);
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

TEST_F(Test__ROCmKernelBaseDeviceContext, GetStream_ReturnsNullWithoutContext)
{
    EXPECT_EQ(kernel_->getStream(), nullptr);
}

TEST_F(Test__ROCmKernelBaseDeviceContext, GetStream_ReturnsContextStream)
{
    void *expected_stream = reinterpret_cast<void *>(0xFEDCBA);
    mock_ctx_->setMockStream(expected_stream);
    kernel_->setDeviceContext(mock_ctx_.get());

    EXPECT_EQ(kernel_->getStream(), expected_stream);
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
