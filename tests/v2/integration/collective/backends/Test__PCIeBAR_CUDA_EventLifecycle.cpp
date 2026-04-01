/**
 * @file Test__PCIeBAR_CUDA_EventLifecycle.cpp
 * @brief Minimal test to reproduce CUDA event "context is destroyed" (error 709)
 *
 * This test isolates the issue where cudaEventQuery returns error 709 when:
 * 1. CUDA event is created on the main thread
 * 2. Event wait is attempted from PCIeBARBackend's worker thread
 * 3. The event was created in a context that the worker thread can't access
 *
 * The key insight is that CUDA events are bound to contexts, and querying an
 * event from a different context (even on the same device) fails.
 *
 * @author David Sanftenberg
 * @date January 2026
 */

#include <gtest/gtest.h>
#include <memory>
#include <thread>
#include <chrono>

#include "backends/ComputeBackend.h"
#include "backends/BackendManager.h"
#include "backends/DeviceId.h"
#include "collective/backends/PCIeBARBackend.h"
#include "collective/BackendRouter.h"
#include "collective/DeviceGroup.h"
#include "tensors/TensorClasses.h"
#include "utils/Logger.h"

#ifdef HAVE_CUDA
#include <cuda_runtime.h>
#include <cuda.h>
#endif

#if defined(HAVE_CUDA) && defined(HAVE_ROCM)

using namespace llaminar2;

class Test__PCIeBAR_CUDA_EventLifecycle : public ::testing::Test
{
protected:
    void SetUp() override
    {
        // Initialize DeviceManager
        DeviceManager::instance().initialize(-1);

        // Check for CUDA GPUs
        auto *cuda_backend = getCUDABackend();
        if (!cuda_backend || cuda_backend->deviceCount() < 1)
        {
            GTEST_SKIP() << "No CUDA GPU available";
        }

        // Check for ROCm GPUs (needed for heterogeneous scenario)
        auto *rocm_backend = getROCmBackend();
        if (!rocm_backend || rocm_backend->deviceCount() < 1)
        {
            GTEST_SKIP() << "No ROCm GPU available (needed for PCIeBAR heterogeneous)";
        }

        cuda_backend_ = cuda_backend;
        rocm_backend_ = rocm_backend;

        // Initialize PCIeBAR backend for heterogeneous tests
        initializePCIeBAR();
    }

    void TearDown() override
    {
        // Shutdown PCIeBAR if we initialized it
        if (pcie_backend_)
        {
            pcie_backend_->shutdown();
            pcie_backend_.reset();
        }
    }

    void initializePCIeBAR()
    {
        // Create PCIeBARBackend via factory
        pcie_backend_ = CollectiveBackendFactory::create(CollectiveBackendType::PCIE_BAR);

        if (!pcie_backend_)
        {
            LOG_WARN("[Test] Failed to create PCIeBARBackend - may not be compiled with HAVE_CUDA && HAVE_ROCM");
            pcie_initialized_ = false;
            return;
        }

        // Create device group with CUDA:0 and ROCm:0
        DeviceGroup group;
        group.name = "test_group";
        group.devices.push_back(DeviceId::cuda(0));
        group.devices.push_back(DeviceId::rocm(0));

        LOG_INFO("[Test] Initializing PCIeBAR backend for test with group: " << group.name);
        if (!pcie_backend_->initialize(group))
        {
            LOG_WARN("[Test] Failed to initialize PCIeBAR backend");
            pcie_backend_.reset();
            pcie_initialized_ = false;
        }
        else
        {
            LOG_INFO("[Test] PCIeBAR backend initialized successfully");
            pcie_initialized_ = true;
        }
    }

    IBackend *cuda_backend_ = nullptr;
    IBackend *rocm_backend_ = nullptr;
    std::unique_ptr<ICollectiveBackend> pcie_backend_;
    bool pcie_initialized_ = false;
};

/**
 * @test Direct event creation and query on main thread
 *
 * This should work - event created and queried in same context.
 */
TEST_F(Test__PCIeBAR_CUDA_EventLifecycle, DirectEventOnMainThread)
{
    // Create event on CUDA device 0
    void *event = cuda_backend_->createEvent(0);
    ASSERT_NE(event, nullptr) << "Failed to create CUDA event";

    // Record event (puts it in pending state)
    ASSERT_TRUE(cuda_backend_->recordEvent(event, 0)) << "Failed to record event";

    // Query should succeed (either returns success or not-ready)
    bool wait_result = cuda_backend_->waitForEvent(event, 0);
    EXPECT_TRUE(wait_result) << "Event wait failed on main thread";

    // Cleanup
    cuda_backend_->destroyEvent(event, 0);
}

/**
 * @test Event created on main thread, queried via PCIeBAR worker
 *
 * This is the problematic scenario - the event is created in the main thread's
 * CUDA context, but we try to query it from the PCIeBAR worker thread which
 * has its own context.
 */
TEST_F(Test__PCIeBAR_CUDA_EventLifecycle, EventCreatedOnMainQueriedViaWorker)
{
    if (!pcie_initialized_)
    {
        GTEST_SKIP() << "PCIeBAR backend not initialized";
    }

    auto *pcie_backend = PCIeBARBackend::getInstance();
    if (!pcie_backend || !pcie_backend->isPCIeBarActive())
    {
        GTEST_SKIP() << "PCIeBAR not active";
    }

    // Create event on CUDA device 0 from main thread
    LOG_INFO("[Test] Creating CUDA event on main thread");
    void *event = cuda_backend_->createEvent(0);
    ASSERT_NE(event, nullptr) << "Failed to create CUDA event";

    // Record event
    LOG_INFO("[Test] Recording CUDA event on main thread");
    ASSERT_TRUE(cuda_backend_->recordEvent(event, 0)) << "Failed to record event";

    // Try to wait via PCIeBAR worker thread - this is where error 709 occurs
    LOG_INFO("[Test] Attempting to wait for event via PCIeBAR worker thread");
    bool worker_result = pcie_backend->waitForCUDAEvent(event, 0);

    // This is the failing case - document the current behavior
    if (!worker_result)
    {
        LOG_WARN("[Test] Worker thread event wait FAILED - this is the bug we're investigating");
        // For now, just note the failure but don't fail the test
        // This helps us confirm we've reproduced the issue
        std::cout << "REPRODUCED: Event created on main thread cannot be queried from worker thread\n";
    }
    else
    {
        std::cout << "UNEXPECTED: Worker thread event wait succeeded!\n";
    }

    // Cleanup
    cuda_backend_->destroyEvent(event, 0);
}

/**
 * @test Simulate the heterogeneous scenario more completely
 *
 * In the real heterogeneous case:
 * 1. Main thread creates inference runners for both CUDA and ROCm
 * 2. Tensors are created with device_completion_event_ on their respective devices
 * 3. When ROCm executor needs CUDA tensor data, it tries to wait for CUDA event
 * 4. This wait is routed through PCIeBAR worker, which fails
 */
TEST_F(Test__PCIeBAR_CUDA_EventLifecycle, SimulateHeterogeneousScenario)
{
    if (!pcie_initialized_)
    {
        GTEST_SKIP() << "PCIeBAR backend not initialized";
    }

    auto *pcie_backend = PCIeBARBackend::getInstance();
    if (!pcie_backend || !pcie_backend->isPCIeBarActive())
    {
        GTEST_SKIP() << "PCIeBAR not active";
    }

    // Create a simple FP32 tensor that will be used on CUDA
    auto tensor = std::make_shared<FP32Tensor>(std::vector<size_t>{1024});
    ASSERT_NE(tensor, nullptr);

    // Allocate on CUDA device 0
    DeviceId cuda_device = DeviceId::cuda(0);
    ASSERT_TRUE(tensor->ensureOnDevice(cuda_device)) << "Failed to allocate tensor on CUDA";

    // Mark device dirty with event - this is what mark_device_dirty_with_event does
    tensor->transitionTo(TensorCoherenceState::DEVICE_AUTHORITATIVE);

    // Now simulate what happens when ROCm executor needs this tensor's data:
    // It calls ensureOnHost() which waits for the CUDA event
    LOG_INFO("[Test] Simulating ROCm executor waiting for CUDA tensor");

    // The tensor should have a completion event now
    // ensureOnHost() will try to wait for it via PCIeBAR worker
    const float *host_data = tensor->data();

    // If we get here without crash/hang, document what happened
    if (host_data != nullptr)
    {
        std::cout << "ensureOnHost() returned non-null: " << host_data[0] << std::endl;
    }
    else
    {
        std::cout << "ensureOnHost() returned nullptr!\n";
    }
}

/**
 * @test Verify that events created BY the worker thread work correctly
 *
 * If the worker thread creates the event itself, it should be in the worker's
 * context and queryable from the worker.
 *
 * NOTE: This test cannot directly use submitCUDAWork (it's private).
 * Instead, we verify through the public waitForCUDAEvent API.
 */
TEST_F(Test__PCIeBAR_CUDA_EventLifecycle, DISABLED_EventCreatedByWorkerThread)
{
    // This test requires access to private submitCUDAWork method.
    // Disabled for now - the key diagnostic is CheckEventContextAffinity.
    GTEST_SKIP() << "Test requires access to private PCIeBARBackend::submitCUDAWork";
}

/**
 * @test Check context affinity of CUDA events
 *
 * This test explicitly checks if an event can be queried from different
 * contexts/threads.
 */
TEST_F(Test__PCIeBAR_CUDA_EventLifecycle, CheckEventContextAffinity)
{
    // Set device and get current context on main thread
    cudaSetDevice(0);

    CUcontext main_ctx = nullptr;
    CUresult cu_err = cuCtxGetCurrent(&main_ctx);
    ASSERT_EQ(cu_err, CUDA_SUCCESS) << "Failed to get main thread context";
    LOG_INFO("[Test] Main thread context: " << (void *)main_ctx);

    // Create event on main thread
    cudaEvent_t event;
    cudaError_t err = cudaEventCreate(&event);
    ASSERT_EQ(err, cudaSuccess) << "Failed to create event";

    // Record event
    err = cudaEventRecord(event, 0);
    ASSERT_EQ(err, cudaSuccess) << "Failed to record event";

    // Query on main thread - should work
    err = cudaEventQuery(event);
    LOG_INFO("[Test] Main thread query: " << err << " (" << cudaGetErrorString(err) << ")");
    EXPECT_TRUE(err == cudaSuccess || err == cudaErrorNotReady);

    // Now query from a different thread
    std::thread query_thread([&]()
                             {
        // Set device on this thread too
        cudaSetDevice(0);
        
        // Get this thread's context
        CUcontext thread_ctx = nullptr;
        CUresult cu_err = cuCtxGetCurrent(&thread_ctx);
        LOG_INFO("[QueryThread] Context: " << (void*)thread_ctx 
                  << " (main was: " << (void*)main_ctx << ")");
        
        // Try to query the event
        cudaError_t err = cudaEventQuery(event);
        LOG_INFO("[QueryThread] Event query: " << err << " (" << cudaGetErrorString(err) << ")");
        
        if (err == 709)
        {
            std::cout << "CONFIRMED: Event query from different thread returns error 709\n";
            std::cout << "This happens because each thread has its own CUDA context!\n";
        }
        else if (err == cudaSuccess || err == cudaErrorNotReady)
        {
            std::cout << "Thread query succeeded - contexts may be the same\n";
        } });
    query_thread.join();

    // Cleanup
    cudaEventDestroy(event);
}

#endif // HAVE_CUDA && HAVE_ROCM
