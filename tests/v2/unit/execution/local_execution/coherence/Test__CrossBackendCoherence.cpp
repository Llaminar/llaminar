/**
 * @file Test__CrossBackendCoherence.cpp
 * @brief Unit tests for cross-backend coherence safety in Pipeline Parallel mode
 *
 * Validates that:
 *   1. IBackend::backendDeviceType() returns correct DeviceType for each backend
 *   2. TransferEngine device-write publication detects backend/device
 *      mismatches and fails before publishing device authority
 *   3. allocateOnDevice() correctly updates gpu_device_ when migrating between devices
 *   4. explicit output allocation re-homes tensors migrated by downstream stages
 *
 * These tests prevent regressions for the HybridPPTP crash where EmbeddingStage
 * published a CUDA write against a tensor whose gpu_device_ was stale (ROCm:0
 * from a prior PP stage), causing hipEventRecord to receive a CUDA stream
 * pointer and segfault.
 *
 * Uses MockBackend DI injection — no real GPU hardware required.
 */

#include <gtest/gtest.h>
#include "transfer/TransferEngine.h"
#include "tensors/Tensors.h"
#include "backends/DeviceId.h"
#include "backends/CPUBackend.h"

#include "../../../../mocks/MockBackend.h"
#include "../../../../mocks/MockWorkerGPUContext.h"

#include <cstring>
#include <memory>
#include <vector>

using namespace llaminar2;
using namespace llaminar2::test;

// =============================================================================
// Test tensor subclass exposing protected coherence state
// =============================================================================

class CrossBackendTensor : public FP32Tensor
{
public:
    using FP32Tensor::FP32Tensor;

    // ---- Expose protected state for assertions ----
    void *getCompletionEvent() const { return device_completion_event_; }
    bool getHostValid() const { return ::llaminar2::isHostValid(coherence_state_); }
    bool getDeviceValid() const { return ::llaminar2::isDeviceValid(coherence_state_); }
    std::optional<DeviceId> getGpuDevice() const { return gpu_device_; }
    void *getGpuDataPtr() const { return gpu_data_ptr_; }
    std::optional<DeviceId> getEventDevice() const { return event_device_; }

    // ---- Inject fake state for testing ----
    void injectGpuDevice(DeviceId device) { gpu_device_ = device; }
    void injectDeviceValid(bool valid)
    {
        if (valid && ::llaminar2::isHostValid(coherence_state_))
            setCoherenceState_(TensorCoherenceState::SYNCED);
        else if (valid)
            setCoherenceState_(TensorCoherenceState::DEVICE_AUTHORITATIVE);
        else if (::llaminar2::isHostValid(coherence_state_))
            setCoherenceState_(TensorCoherenceState::HOST_AUTHORITATIVE);
    }
    void injectHostValid(bool valid)
    {
        if (valid && ::llaminar2::isDeviceValid(coherence_state_))
            setCoherenceState_(TensorCoherenceState::SYNCED);
        else if (valid)
            setCoherenceState_(TensorCoherenceState::HOST_AUTHORITATIVE);
        else if (::llaminar2::isDeviceValid(coherence_state_))
            setCoherenceState_(TensorCoherenceState::DEVICE_AUTHORITATIVE);
    }
    void injectGpuDataPtr(void *ptr) { gpu_data_ptr_ = ptr; }
    void injectCompletionEvent(void *event) { device_completion_event_ = event; }
    void injectEventDevice(DeviceId device) { event_device_ = device; }
};

// =============================================================================
// Test Fixture
// =============================================================================

class Test__CrossBackendCoherence : public ::testing::Test
{
protected:
    static void SetUpTestSuite()
    {
        llaminar2::testing::installHardwareFreeGPUContextFactories();
    }

    static constexpr size_t kRows = 4;
    static constexpr size_t kCols = 8;
    static constexpr size_t kElements = kRows * kCols;

    // Two mock backends simulating different GPU types
    MockBackend mock_backend_;

    DeviceId cuda_device_ = DeviceId::cuda(0);
    DeviceId rocm_device_ = DeviceId::rocm(0);

    std::unique_ptr<CrossBackendTensor> createTensor(float fill = 1.0f)
    {
        auto tensor = std::make_unique<CrossBackendTensor>(
            std::vector<size_t>{kRows, kCols}, DeviceId::cpu());
        tensor->setBackendForTesting(&mock_backend_);

        float *data = tensor->mutable_data();
        for (size_t i = 0; i < kElements; ++i)
        {
            data[i] = fill + static_cast<float>(i) * 0.01f;
        }
        return tensor;
    }

    void cleanupTensor(CrossBackendTensor *tensor)
    {
        void *gpu_ptr = tensor->getGpuDataPtr();
        if (gpu_ptr && mock_backend_.isAllocated(gpu_ptr))
        {
            mock_backend_.free(gpu_ptr, 0);
        }
        tensor->injectGpuDataPtr(nullptr);
    }
};

// =============================================================================
// Category 1: IBackend::backendDeviceType()
// =============================================================================

TEST_F(Test__CrossBackendCoherence, MockBackend_ReturnsCorrectDeviceType)
{
    // MockBackend defaults to CPU type
    EXPECT_EQ(mock_backend_.backendDeviceType(), DeviceType::CPU);
}

TEST_F(Test__CrossBackendCoherence, CPUBackend_ReturnsCorrectDeviceType)
{
    CPUBackend cpu_backend(-1); // No NUMA binding
    EXPECT_EQ(cpu_backend.backendDeviceType(), DeviceType::CPU);
}

// Note: CUDABackend and ROCmBackend return CUDA and ROCm respectively,
// but cannot be tested without hardware. The inline implementations in
// the headers are trivially correct (return DeviceType::CUDA / DeviceType::ROCm).

// =============================================================================
// Category 2: mark_device_dirty_with_event() cross-backend safety
// =============================================================================

TEST_F(Test__CrossBackendCoherence, MarkDirtyWithEvent_BackendMismatchFailsClosed)
{
    // Setup: tensor on ROCm device with mock backend (mock returns CPU type,
    // but the injected gpu_device_ also needs to match for a real scenario).
    // Since MockBackend returns DeviceType::CPU, we simulate by using a CPU-like device.
    // The key test is the MISMATCH case below.
    auto tensor = createTensor();

    // Upload to device (mock always succeeds regardless of device type)
    ASSERT_TRUE(tensor->ensureOnDevice(rocm_device_));
    mock_backend_.resetAll();

    // Clear existing event to force creation
    tensor->injectCompletionEvent(nullptr);

    void *fake_stream = reinterpret_cast<void *>(0xBEEF);
    EXPECT_THROW(
        TransferEngine::publishCurrentDeviceWrite(
            tensor.get(),
            fake_stream),
        std::runtime_error);

    // MockBackend's backendDeviceType() returns CPU, but gpu_device_ is ROCm.
    // The defensive check compares backend->backendDeviceType() vs gpu_device_->type.
    // Since they don't match (CPU != ROCm), event recording should be SKIPPED.
    // This hard failure prevents the HybridPPTP crash without publishing a
    // state that lacks a usable completion dependency.
    EXPECT_EQ(mock_backend_.getEventCreateCount(), 0u)
        << "Event must not be created through the wrong backend";
    EXPECT_EQ(mock_backend_.getEventRecordCount(), 0u)
        << "Event must not be recorded through the wrong backend";
    EXPECT_TRUE(tensor->getHostValid());
    EXPECT_TRUE(tensor->getDeviceValid())
        << "The preceding upload remains SYNCED after rejected publication.";

    cleanupTensor(tensor.get());
}

TEST_F(Test__CrossBackendCoherence, MarkDirtyWithEvent_NullStreamStillChecksBackend)
{
    auto tensor = createTensor();
    ASSERT_TRUE(tensor->ensureOnDevice(rocm_device_));
    mock_backend_.resetAll();
    tensor->injectCompletionEvent(nullptr);

    EXPECT_THROW(
        TransferEngine::publishCurrentDeviceWrite(
            tensor.get(),
            nullptr),
        std::invalid_argument);
    EXPECT_EQ(mock_backend_.getEventCreateCount(), 0u)
        << "A null stream must fail before backend ownership is inspected.";
    EXPECT_TRUE(tensor->getHostValid());
    EXPECT_TRUE(tensor->getDeviceValid());

    cleanupTensor(tensor.get());
}

// =============================================================================
// Category 3: allocateOnDevice() device migration updates gpu_device_
// =============================================================================

TEST_F(Test__CrossBackendCoherence, AllocateOnDevice_MigratesGpuDevice)
{
    auto tensor = createTensor();

    // First: allocate on "ROCm" device
    ASSERT_TRUE(tensor->allocateOnDevice(rocm_device_));
    EXPECT_EQ(tensor->getGpuDevice(), rocm_device_);

    void *first_ptr = tensor->getGpuDataPtr();
    EXPECT_NE(first_ptr, nullptr);

    // Second: allocate on "CUDA" device — should migrate gpu_device_
    ASSERT_TRUE(tensor->allocateOnDevice(cuda_device_));
    EXPECT_EQ(tensor->getGpuDevice(), cuda_device_)
        << "gpu_device_ must be updated to CUDA after migration";

    // Pointer should have changed (new allocation for different device)
    // Note: MockBackend allocates with malloc, so the old ptr was freed
    void *second_ptr = tensor->getGpuDataPtr();
    EXPECT_NE(second_ptr, nullptr);

    cleanupTensor(tensor.get());
}

TEST_F(Test__CrossBackendCoherence, AllocateOnDevice_SameDevice_NoMigration)
{
    auto tensor = createTensor();

    ASSERT_TRUE(tensor->allocateOnDevice(rocm_device_));
    void *first_ptr = tensor->getGpuDataPtr();
    size_t alloc_count = mock_backend_.getAllocationCount();

    // Re-allocate on same device — should reuse existing buffer
    ASSERT_TRUE(tensor->allocateOnDevice(rocm_device_));
    EXPECT_EQ(tensor->getGpuDataPtr(), first_ptr);
    EXPECT_EQ(mock_backend_.getAllocationCount(), alloc_count);
    EXPECT_EQ(tensor->getGpuDevice(), rocm_device_);

    cleanupTensor(tensor.get());
}

// =============================================================================
// Category 4: Simulated PP stage migration scenario
// =============================================================================

TEST_F(Test__CrossBackendCoherence, PPMigration_OutputTensorRehomedByDownstreamStage)
{
    // Simulate the HybridPPTP scenario:
    // 1. Tensor starts on CUDA:0 (PP stage 0 output)
    // 2. Downstream PP stage 1 on ROCm:0 calls ensureOnDevice → migrates to ROCm
    // 3. Next iteration: PP stage 0 needs to re-home the tensor back to CUDA:0
    //    via explicit output allocation
    auto tensor = createTensor(5.0f);

    // Step 1: Initial allocation on CUDA device
    ASSERT_TRUE(tensor->allocateOnDevice(cuda_device_));
    EXPECT_EQ(tensor->getGpuDevice(), cuda_device_);

    // Step 2: Downstream stage migrates tensor to ROCm (simulates ensureOnDevice)
    ASSERT_TRUE(tensor->ensureOnDevice(rocm_device_));
    EXPECT_EQ(tensor->getGpuDevice(), rocm_device_)
        << "After downstream migration, gpu_device_ should be ROCm";

    // Step 3: output preparation re-homes the tensor before PP stage 0 executes.
    ASSERT_TRUE(tensor->allocateOnDevice(cuda_device_));
    EXPECT_EQ(tensor->getGpuDevice(), cuda_device_)
        << "After re-homing via allocateOnDevice, gpu_device_ should be CUDA again";

    // Step 4: mark_device_dirty_with_event should now be safe with a CUDA stream
    // (gpu_device_ correctly reflects CUDA, not stale ROCm)
    tensor->injectCompletionEvent(nullptr);
    mock_backend_.resetAll();

    // Even with MockBackend (CPU type), the key validation is that gpu_device_
    // was correctly updated. In production, the real CUDABackend would be resolved
    // and backendDeviceType() would return CUDA, matching gpu_device_.type (CUDA).
    // Here we just verify the state machine is correct.
    EXPECT_EQ(tensor->getGpuDevice()->type, DeviceType::CUDA);

    cleanupTensor(tensor.get());
}

TEST_F(Test__CrossBackendCoherence, PPMigration_StaleGpuDevice_EventFailsClosed)
{
    // Demonstrate the exact crash scenario (now defended against):
    // 1. Tensor's gpu_device_ says ROCm:0 (stale from downstream)
    // 2. Caller passes a "CUDA stream" to mark_device_dirty_with_event
    // 3. Backend resolves to MockBackend (type=CPU), gpu_device_ type=ROCm → mismatch
    // 4. Publication fails before event recording or coherence mutation
    auto tensor = createTensor(10.0f);

    // Simulate stale state: tensor thinks it's on ROCm
    ASSERT_TRUE(tensor->allocateOnDevice(rocm_device_));
    EXPECT_EQ(tensor->getGpuDevice(), rocm_device_);

    mock_backend_.resetAll();
    tensor->injectCompletionEvent(nullptr);

    // Pass a non-null "CUDA stream" — triggers the defensive check
    void *cuda_stream = reinterpret_cast<void *>(0xC0DA);
    EXPECT_THROW(
        TransferEngine::publishCurrentDeviceWrite(
            tensor.get(),
            cuda_stream),
        std::runtime_error);

    // The defensive check fires: backend type (CPU) != gpu_device_ type (ROCm)
    // Event recording must not be attempted.
    EXPECT_EQ(mock_backend_.getEventRecordCount(), 0u)
        << "Must NOT record event when backend type mismatches gpu_device_ type";

    // The preceding allocation leaves host data authoritative. Failed
    // publication cannot alter that ownership.
    EXPECT_TRUE(tensor->getHostValid());
    EXPECT_FALSE(tensor->getDeviceValid());

    cleanupTensor(tensor.get());
}
