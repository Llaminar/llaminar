/**
 * @file Test__CoherenceProtocol.cpp
 * @brief Comprehensive unit tests for the GPU tensor coherence protocol
 *
 * Tests all major data paths in TensorBase coherence using MockBackend DI:
 *   1. ensureOnDevice() H2D round-trip (allocation, transfer, idempotency)
 *   2. data() accessor triggers coherence (device-dirty → sync → D2H)
 *   3. allocateOnDevice() without H2D (allocation-only path)
 *   4. ensureOnHost() no-op when host-valid
 *   5. mark_device_dirty_with_event() event lifecycle
 *   6. Full coherence state machine cycle
 *   7. Failure paths (allocation, H2D, D2H failures)
 *
 * All tests use DeviceId::rocm(0) to avoid PCIeBAR proxy in waitForEventWithProxy.
 */

#include <gtest/gtest.h>
#include "tensors/Tensors.h"
#include "backends/DeviceId.h"

#include "../../../../mocks/MockBackend.h"

#include <cstring>
#include <memory>
#include <numeric>
#include <vector>

using namespace llaminar2;
using namespace llaminar2::test;

// =============================================================================
// MockCoherenceTensor: FP32Tensor subclass exposing protected coherence state
// =============================================================================

class CoherenceProtocolTensor : public FP32Tensor
{
public:
    using FP32Tensor::FP32Tensor;

    // ---- Expose protected state for assertions ----

    void *getCompletionEvent() const { return device_completion_event_; }
    bool getHostValid() const { return ::llaminar2::isHostValid(coherence_state_); }
    bool getDeviceValid() const { return ::llaminar2::isDeviceValid(coherence_state_); }
    std::optional<DeviceId> getGpuDevice() const { return gpu_device_; }
    std::optional<DeviceId> getAuthoritativeDevice() const { return authoritative_device_; }
    void *getGpuDataPtr() const { return gpu_data_ptr_; }

    // ---- Inject fake state for testing ----

    void injectCompletionEvent(void *event) { device_completion_event_ = event; }
    void injectGpuDevice(DeviceId device) { gpu_device_ = device; }
    void injectDeviceValid(bool valid)
    {
        if (valid)
        {
            if (::llaminar2::isHostValid(coherence_state_))
                setCoherenceState_(TensorCoherenceState::SYNCED);
            else
                setCoherenceState_(TensorCoherenceState::DEVICE_AUTHORITATIVE);
        }
        else
        {
            if (::llaminar2::isHostValid(coherence_state_))
                setCoherenceState_(TensorCoherenceState::HOST_AUTHORITATIVE);
            else
                setCoherenceState_(TensorCoherenceState::INVALID);
        }
    }
    void injectHostValid(bool valid)
    {
        if (valid)
        {
            if (::llaminar2::isDeviceValid(coherence_state_))
                setCoherenceState_(TensorCoherenceState::SYNCED);
            else
                setCoherenceState_(TensorCoherenceState::HOST_AUTHORITATIVE);
        }
        else
        {
            if (::llaminar2::isDeviceValid(coherence_state_))
                setCoherenceState_(TensorCoherenceState::DEVICE_AUTHORITATIVE);
            else
                setCoherenceState_(TensorCoherenceState::INVALID);
        }
    }

    void injectGpuDataPtr(void *ptr) { gpu_data_ptr_ = ptr; }
};

// =============================================================================
// Test Fixture
// =============================================================================

class Test__CoherenceProtocol : public ::testing::Test
{
protected:
    static constexpr size_t kRows = 4;
    static constexpr size_t kCols = 8;
    static constexpr size_t kElements = kRows * kCols;
    static constexpr size_t kBytes = kElements * sizeof(float);

    // Use ROCm device to avoid PCIeBAR proxy in waitForEventWithProxy
    // Use ROCm device to avoid PCIeBAR proxy in waitForEventWithProxy
    DeviceId device_ = DeviceId::rocm(0);

    // Mock backend configured as ROCm type to match device_
    MockBackend mock_backend_{DeviceType::ROCm};

    /// Create a tensor with known data pattern and inject mock backend
    std::unique_ptr<CoherenceProtocolTensor> createTensor(float fill_value = 1.0f)
    {
        auto tensor = std::make_unique<CoherenceProtocolTensor>(
            std::vector<size_t>{kRows, kCols}, DeviceId::cpu());
        tensor->setBackendForTesting(&mock_backend_);

        // Fill with known pattern
        float *data = tensor->mutable_data();
        for (size_t i = 0; i < kElements; ++i)
        {
            data[i] = fill_value + static_cast<float>(i) * 0.01f;
        }

        return tensor;
    }

    /// Safe cleanup: free mock allocation and null out gpu_data_ptr before destruction
    void cleanupTensor(CoherenceProtocolTensor *tensor)
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
// Category 1: ensureOnDevice() H2D Round-Trip
// =============================================================================

TEST_F(Test__CoherenceProtocol, EnsureOnDevice_AllocatesAndTransfers)
{
    auto tensor = createTensor(1.0f);

    // Pre-conditions
    EXPECT_EQ(mock_backend_.getAllocationCount(), 0u);
    EXPECT_EQ(mock_backend_.getH2DCount(), 0u);

    // Act
    bool ok = tensor->ensureOnDevice(device_);
    ASSERT_TRUE(ok);

    // Verify allocation happened
    EXPECT_GE(mock_backend_.getAllocationCount(), 1u);
    EXPECT_GE(mock_backend_.getTotalAllocatedBytes(), kBytes);

    // Verify H2D transfer happened
    EXPECT_GE(mock_backend_.getH2DCount(), 1u);

    // Verify coherence flags
    EXPECT_TRUE(tensor->getDeviceValid());
    EXPECT_TRUE(tensor->getHostValid()); // Host data wasn't invalidated
    EXPECT_EQ(tensor->getGpuDevice(), device_);

    // Verify data integrity: read back from mock "device" memory
    void *gpu_ptr = tensor->getGpuDataPtr();
    ASSERT_NE(gpu_ptr, nullptr);

    const float *device_data = static_cast<const float *>(gpu_ptr);
    for (size_t i = 0; i < kElements; ++i)
    {
        EXPECT_FLOAT_EQ(device_data[i], 1.0f + static_cast<float>(i) * 0.01f)
            << "Mismatch at index " << i;
    }

    cleanupTensor(tensor.get());
}

TEST_F(Test__CoherenceProtocol, EnsureOnDevice_IdempotentSecondCall)
{
    auto tensor = createTensor(2.0f);

    // First call: allocate + H2D
    ASSERT_TRUE(tensor->ensureOnDevice(device_));
    size_t alloc_count_after_first = mock_backend_.getAllocationCount();
    size_t h2d_count_after_first = mock_backend_.getH2DCount();

    EXPECT_GE(alloc_count_after_first, 1u);
    EXPECT_GE(h2d_count_after_first, 1u);

    // Second call: should be a no-op (device already valid on same device)
    ASSERT_TRUE(tensor->ensureOnDevice(device_));

    EXPECT_EQ(mock_backend_.getAllocationCount(), alloc_count_after_first);
    EXPECT_EQ(mock_backend_.getH2DCount(), h2d_count_after_first);

    cleanupTensor(tensor.get());
}

TEST_F(Test__CoherenceProtocol, EnsureOnDevice_ReUploadsAfterHostModification)
{
    auto tensor = createTensor(3.0f);

    // First upload
    ASSERT_TRUE(tensor->ensureOnDevice(device_));
    size_t h2d_after_first = mock_backend_.getH2DCount();

    // Modify host data (mutable_data() sets host as authoritative)
    float *host_data = tensor->mutable_data();
    for (size_t i = 0; i < kElements; ++i)
    {
        host_data[i] = 99.0f;
    }

    // Second upload: should detect host has newer data and re-upload
    ASSERT_TRUE(tensor->ensureOnDevice(device_));
    EXPECT_GT(mock_backend_.getH2DCount(), h2d_after_first);

    // Verify new data was uploaded
    const float *device_data = static_cast<const float *>(tensor->getGpuDataPtr());
    for (size_t i = 0; i < kElements; ++i)
    {
        EXPECT_FLOAT_EQ(device_data[i], 99.0f)
            << "Re-uploaded data mismatch at index " << i;
    }

    cleanupTensor(tensor.get());
}

// =============================================================================
// Category 2: data() Accessor Triggers Coherence
// =============================================================================

TEST_F(Test__CoherenceProtocol, DataAccessor_TriggersD2HWhenDeviceDirty)
{
    auto tensor = createTensor(5.0f);

    // Upload to device
    ASSERT_TRUE(tensor->ensureOnDevice(device_));

    // Simulate GPU modifying data: write new values to mock device memory
    float *gpu_mem = static_cast<float *>(tensor->getGpuDataPtr());
    for (size_t i = 0; i < kElements; ++i)
    {
        gpu_mem[i] = 42.0f + static_cast<float>(i);
    }

    // Mark as device-dirty (flags only, to test the no-event sync path)
    tensor->mark_device_dirty_flags_only();

    EXPECT_TRUE(tensor->getDeviceValid());
    EXPECT_FALSE(tensor->getHostValid());

    mock_backend_.resetAll();

    // Access data() — should trigger ensureOnHost → sync + D2H
    const float *result = tensor->data();
    ASSERT_NE(result, nullptr);

    // Verify sync happened (full sync since no event)
    EXPECT_GE(mock_backend_.getSyncCount(), 1u);

    // Verify D2H transfer happened
    EXPECT_GE(mock_backend_.getD2HCount(), 1u);

    // Verify data was correctly transferred back
    for (size_t i = 0; i < kElements; ++i)
    {
        EXPECT_FLOAT_EQ(result[i], 42.0f + static_cast<float>(i))
            << "D2H data mismatch at index " << i;
    }

    // Host should now be valid
    EXPECT_TRUE(tensor->getHostValid());

    cleanupTensor(tensor.get());
}

TEST_F(Test__CoherenceProtocol, DataAccessor_NoTransferWhenHostValid)
{
    auto tensor = createTensor(7.0f);

    // Tensor starts with host_valid = true (never uploaded to device)
    EXPECT_TRUE(tensor->getHostValid());

    mock_backend_.resetAll();

    // Access data() — should NOT trigger any sync or D2H
    const float *result = tensor->data();
    ASSERT_NE(result, nullptr);

    EXPECT_EQ(mock_backend_.getSyncCount(), 0u);
    EXPECT_EQ(mock_backend_.getD2HCount(), 0u);

    // Data should still be the original values
    EXPECT_FLOAT_EQ(result[0], 7.0f);
}

// =============================================================================
// Category 3: allocateOnDevice() Without H2D
// =============================================================================

TEST_F(Test__CoherenceProtocol, AllocateOnDevice_AllocatesWithoutTransfer)
{
    auto tensor = createTensor(10.0f);

    EXPECT_EQ(mock_backend_.getAllocationCount(), 0u);
    EXPECT_EQ(mock_backend_.getH2DCount(), 0u);

    // allocateOnDevice should only allocate, not transfer
    bool ok = tensor->allocateOnDevice(device_);
    ASSERT_TRUE(ok);

    EXPECT_GE(mock_backend_.getAllocationCount(), 1u);
    EXPECT_EQ(mock_backend_.getH2DCount(), 0u);

    // Device should NOT be marked valid (no data transferred)
    EXPECT_FALSE(tensor->getDeviceValid());

    // GPU data ptr should be set
    EXPECT_NE(tensor->getGpuDataPtr(), nullptr);

    // Host should still be valid
    EXPECT_TRUE(tensor->getHostValid());

    cleanupTensor(tensor.get());
}

TEST_F(Test__CoherenceProtocol, AllocateOnDevice_IdempotentSecondCall)
{
    auto tensor = createTensor(11.0f);

    ASSERT_TRUE(tensor->allocateOnDevice(device_));
    void *first_ptr = tensor->getGpuDataPtr();
    size_t alloc_count = mock_backend_.getAllocationCount();

    // Second call should reuse existing allocation
    ASSERT_TRUE(tensor->allocateOnDevice(device_));
    EXPECT_EQ(tensor->getGpuDataPtr(), first_ptr);
    EXPECT_EQ(mock_backend_.getAllocationCount(), alloc_count);

    cleanupTensor(tensor.get());
}

// =============================================================================
// Category 4: ensureOnHost() No-Op When Host Valid
// =============================================================================

TEST_F(Test__CoherenceProtocol, EnsureOnHost_NoOpWhenHostValid)
{
    auto tensor = createTensor(15.0f);

    // Tensor starts with host_valid = true
    EXPECT_TRUE(tensor->getHostValid());

    mock_backend_.resetAll();

    bool ok = tensor->ensureOnHost();
    ASSERT_TRUE(ok);

    // Should NOT have triggered any sync or D2H
    EXPECT_EQ(mock_backend_.getSyncCount(), 0u);
    EXPECT_EQ(mock_backend_.getStreamSyncCount(), 0u);
    EXPECT_EQ(mock_backend_.getD2HCount(), 0u);
}

TEST_F(Test__CoherenceProtocol, EnsureOnHost_SkipsSyncWhenHostValidEvenWithGpuDevice)
{
    auto tensor = createTensor(16.0f);

    // Upload to device (both host and device are now valid)
    ASSERT_TRUE(tensor->ensureOnDevice(device_));
    EXPECT_TRUE(tensor->getHostValid());
    EXPECT_TRUE(tensor->getDeviceValid());

    mock_backend_.resetAll();

    // ensureOnHost when both are valid → no transfer needed
    bool ok = tensor->ensureOnHost();
    ASSERT_TRUE(ok);

    EXPECT_EQ(mock_backend_.getSyncCount(), 0u);
    EXPECT_EQ(mock_backend_.getD2HCount(), 0u);
}

// =============================================================================
// Category 5: mark_device_dirty_with_event() Event Lifecycle
// =============================================================================

TEST_F(Test__CoherenceProtocol, MarkDirtyWithEvent_CreatesAndRecordsEvent)
{
    auto tensor = createTensor(20.0f);

    // Need to set up gpu_device_ so that mark_device_dirty_with_event finds a backend
    ASSERT_TRUE(tensor->ensureOnDevice(device_));
    mock_backend_.resetAll();

    // Pre-condition: may or may not have an event from ensureOnDevice
    // Clear it so we test fresh event creation
    tensor->injectCompletionEvent(nullptr);

    void *fake_stream = reinterpret_cast<void *>(0xBEEF);
    tensor->mark_device_dirty_with_event(fake_stream);

    // Verify event was created
    EXPECT_GE(mock_backend_.getEventCreateCount(), 1u);

    // Verify event was recorded on the stream
    EXPECT_GE(mock_backend_.getEventRecordCount(), 1u);

    // Verify completion event is set on the tensor
    EXPECT_NE(tensor->getCompletionEvent(), nullptr);

    // Verify coherence flags
    EXPECT_TRUE(tensor->getDeviceValid());
    EXPECT_FALSE(tensor->getHostValid()); // Host invalidated

    cleanupTensor(tensor.get());
}

TEST_F(Test__CoherenceProtocol, MarkDirtyWithEvent_ReusesExistingEvent)
{
    auto tensor = createTensor(21.0f);
    ASSERT_TRUE(tensor->ensureOnDevice(device_));
    mock_backend_.resetAll();

    // First call: creates event + records
    tensor->mark_device_dirty_with_event(nullptr);
    size_t creates_after_first = mock_backend_.getEventCreateCount();
    size_t records_after_first = mock_backend_.getEventRecordCount();
    void *event_after_first = tensor->getCompletionEvent();

    EXPECT_GE(creates_after_first, 1u);
    EXPECT_GE(records_after_first, 1u);

    // Second call: should reuse existing event, only record again
    tensor->mark_device_dirty_with_event(nullptr);
    EXPECT_EQ(mock_backend_.getEventCreateCount(), creates_after_first); // No new create
    EXPECT_GT(mock_backend_.getEventRecordCount(), records_after_first); // New record
    EXPECT_EQ(tensor->getCompletionEvent(), event_after_first);         // Same event

    cleanupTensor(tensor.get());
}

TEST_F(Test__CoherenceProtocol, MarkDirtyWithEvent_EventUsedForSyncOnReadback)
{
    auto tensor = createTensor(22.0f);
    ASSERT_TRUE(tensor->ensureOnDevice(device_));

    // Write new data to "device"
    float *gpu_mem = static_cast<float *>(tensor->getGpuDataPtr());
    for (size_t i = 0; i < kElements; ++i)
    {
        gpu_mem[i] = 100.0f + static_cast<float>(i);
    }

    // Mark dirty with event
    tensor->mark_device_dirty_with_event(nullptr);

    mock_backend_.resetAll();

    // Read back — should use event wait, NOT full sync
    const float *result = tensor->data();
    ASSERT_NE(result, nullptr);

    // Event wait should have been called
    EXPECT_GE(mock_backend_.getEventWaitCount(), 1u);
    // D2H transfer should have happened
    EXPECT_GE(mock_backend_.getD2HCount(), 1u);

    // Verify data integrity
    for (size_t i = 0; i < kElements; ++i)
    {
        EXPECT_FLOAT_EQ(result[i], 100.0f + static_cast<float>(i))
            << "Event-synced D2H mismatch at index " << i;
    }

    cleanupTensor(tensor.get());
}

// =============================================================================
// Category 6: Full Coherence State Machine Cycle
// =============================================================================

TEST_F(Test__CoherenceProtocol, FullStateMachineCycle_HostDeviceDirtyReadback)
{
    auto tensor = createTensor(30.0f);

    // --- Step 1: Initial state ---
    EXPECT_TRUE(tensor->getHostValid());
    EXPECT_FALSE(tensor->getDeviceValid());
    EXPECT_EQ(tensor->getGpuDevice(), std::nullopt);
    EXPECT_EQ(tensor->getGpuDataPtr(), nullptr);

    // --- Step 2: Upload to device ---
    ASSERT_TRUE(tensor->ensureOnDevice(device_));
    EXPECT_TRUE(tensor->getHostValid());
    EXPECT_TRUE(tensor->getDeviceValid());
    EXPECT_EQ(tensor->getGpuDevice(), device_);
    EXPECT_NE(tensor->getGpuDataPtr(), nullptr);

    // Verify data integrity after upload
    const float *gpu_data = static_cast<const float *>(tensor->getGpuDataPtr());
    EXPECT_FLOAT_EQ(gpu_data[0], 30.0f);

    // --- Step 3: Mark device dirty (simulating GPU kernel execution) ---
    float *gpu_mem = static_cast<float *>(tensor->getGpuDataPtr());
    for (size_t i = 0; i < kElements; ++i)
    {
        gpu_mem[i] = 77.0f;
    }
    tensor->mark_device_dirty_flags_only();

    EXPECT_TRUE(tensor->getDeviceValid());
    EXPECT_FALSE(tensor->getHostValid());
    EXPECT_TRUE(tensor->getAuthoritativeDevice().has_value());
    EXPECT_EQ(*tensor->getAuthoritativeDevice(), device_);

    // --- Step 4: Read back to host ---
    mock_backend_.resetAll();
    const float *host_data = tensor->data();
    ASSERT_NE(host_data, nullptr);

    // Should have synced and transferred
    EXPECT_GE(mock_backend_.getSyncCount(), 1u); // Full sync (no event)
    EXPECT_GE(mock_backend_.getD2HCount(), 1u);

    // Both should now be valid
    EXPECT_TRUE(tensor->getHostValid());
    EXPECT_TRUE(tensor->getDeviceValid());

    // Verify data correctness
    for (size_t i = 0; i < kElements; ++i)
    {
        EXPECT_FLOAT_EQ(host_data[i], 77.0f)
            << "Full cycle readback mismatch at index " << i;
    }

    // --- Step 5: Modify host and re-upload ---
    float *mutable_host = tensor->mutable_data();
    for (size_t i = 0; i < kElements; ++i)
    {
        mutable_host[i] = 88.0f;
    }

    mock_backend_.resetAll();
    ASSERT_TRUE(tensor->ensureOnDevice(device_));
    EXPECT_GE(mock_backend_.getH2DCount(), 1u);

    const float *re_uploaded = static_cast<const float *>(tensor->getGpuDataPtr());
    for (size_t i = 0; i < kElements; ++i)
    {
        EXPECT_FLOAT_EQ(re_uploaded[i], 88.0f)
            << "Re-upload data mismatch at index " << i;
    }

    cleanupTensor(tensor.get());
}

TEST_F(Test__CoherenceProtocol, FullCycle_WithEventSync)
{
    auto tensor = createTensor(40.0f);

    // Upload
    ASSERT_TRUE(tensor->ensureOnDevice(device_));

    // GPU "computes" new data
    float *gpu_mem = static_cast<float *>(tensor->getGpuDataPtr());
    for (size_t i = 0; i < kElements; ++i)
    {
        gpu_mem[i] = 55.5f;
    }

    // Mark dirty WITH event (full path)
    tensor->mark_device_dirty_with_event(reinterpret_cast<void *>(0xCAFE));

    EXPECT_TRUE(tensor->getDeviceValid());
    EXPECT_FALSE(tensor->getHostValid());
    EXPECT_NE(tensor->getCompletionEvent(), nullptr);

    mock_backend_.resetAll();

    // Read back — should use event wait path
    const float *result = tensor->data();
    ASSERT_NE(result, nullptr);

    // Event-based path should use waitForEvent, not full sync
    EXPECT_GE(mock_backend_.getEventWaitCount(), 1u);
    EXPECT_GE(mock_backend_.getD2HCount(), 1u);

    for (size_t i = 0; i < kElements; ++i)
    {
        EXPECT_FLOAT_EQ(result[i], 55.5f);
    }

    cleanupTensor(tensor.get());
}

// =============================================================================
// Category 7: Failure Paths
// =============================================================================

/// MockBackend subclass that can simulate failures
class FailingMockBackend : public MockBackend
{
public:
    bool fail_allocate = false;
    bool fail_h2d = false;
    bool fail_d2h = false;

    void *allocate(size_t bytes, int device_id) override
    {
        if (fail_allocate)
            return nullptr;
        return MockBackend::allocate(bytes, device_id);
    }

    bool hostToDevice(void *dst, const void *src, size_t bytes, int device_id) override
    {
        if (fail_h2d)
            return false;
        return MockBackend::hostToDevice(dst, src, bytes, device_id);
    }

    bool deviceToHost(void *dst, const void *src, size_t bytes, int device_id) override
    {
        if (fail_d2h)
            return false;
        return MockBackend::deviceToHost(dst, src, bytes, device_id);
    }
};

class Test__CoherenceProtocolFailure : public ::testing::Test
{
protected:
    static constexpr size_t kRows = 4;
    static constexpr size_t kCols = 8;

    DeviceId device_ = DeviceId::rocm(0);
    FailingMockBackend mock_backend_;

    std::unique_ptr<CoherenceProtocolTensor> createTensor()
    {
        auto tensor = std::make_unique<CoherenceProtocolTensor>(
            std::vector<size_t>{kRows, kCols}, DeviceId::cpu());
        tensor->setBackendForTesting(&mock_backend_);

        float *data = tensor->mutable_data();
        for (size_t i = 0; i < kRows * kCols; ++i)
        {
            data[i] = static_cast<float>(i);
        }
        return tensor;
    }

    void cleanupTensor(CoherenceProtocolTensor *tensor)
    {
        void *gpu_ptr = tensor->getGpuDataPtr();
        if (gpu_ptr && mock_backend_.isAllocated(gpu_ptr))
        {
            mock_backend_.free(gpu_ptr, 0);
        }
        tensor->injectGpuDataPtr(nullptr);
    }
};

TEST_F(Test__CoherenceProtocolFailure, AllocationFailure_EnsureOnDeviceReturnsFalse)
{
    auto tensor = createTensor();
    mock_backend_.fail_allocate = true;

    bool ok = tensor->ensureOnDevice(device_);
    EXPECT_FALSE(ok);

    // Tensor should remain in CPU-only state
    EXPECT_TRUE(tensor->getHostValid());
    EXPECT_FALSE(tensor->getDeviceValid());
    EXPECT_EQ(tensor->getGpuDataPtr(), nullptr);
}

TEST_F(Test__CoherenceProtocolFailure, H2DFailure_EnsureOnDeviceReturnsFalse)
{
    auto tensor = createTensor();
    mock_backend_.fail_h2d = true;

    bool ok = tensor->ensureOnDevice(device_);
    EXPECT_FALSE(ok);

    // Host data should still be valid
    EXPECT_TRUE(tensor->getHostValid());

    cleanupTensor(tensor.get());
}

TEST_F(Test__CoherenceProtocolFailure, D2HFailure_EnsureOnHostReturnsFalse)
{
    auto tensor = createTensor();

    // First, successfully upload to device
    ASSERT_TRUE(tensor->ensureOnDevice(device_));

    // Write new data on "device"
    float *gpu_mem = static_cast<float *>(tensor->getGpuDataPtr());
    gpu_mem[0] = 999.0f;

    // Mark dirty so ensureOnHost will attempt D2H
    tensor->mark_device_dirty_flags_only();

    // Now make D2H fail
    mock_backend_.fail_d2h = true;

    bool ok = tensor->ensureOnHost();
    EXPECT_FALSE(ok);

    // Host should NOT be marked valid since D2H failed
    EXPECT_FALSE(tensor->getHostValid());

    cleanupTensor(tensor.get());
}

TEST_F(Test__CoherenceProtocolFailure, AllocationFailure_AllocateOnDeviceReturnsFalse)
{
    auto tensor = createTensor();
    mock_backend_.fail_allocate = true;

    bool ok = tensor->allocateOnDevice(device_);
    EXPECT_FALSE(ok);

    EXPECT_EQ(tensor->getGpuDataPtr(), nullptr);
}
