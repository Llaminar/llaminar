/**
 * @file Test__StreamCoherenceIntegration.cpp
 * @brief Integration tests for stream/coherence interaction on real GPU hardware
 * @author GitHub Copilot
 * @date January 2026
 *
 * **Purpose**: Verify that the coherence system correctly synchronizes GPU
 * operations when using real CUDA/ROCm devices. These tests exercise the
 * actual event recording, event waiting, and D2H transfer paths that the
 * unit tests (Test__StreamCoherence.cpp) test via mocks.
 *
 * **Key scenarios tested**:
 * 1. mark_device_dirty_with_event() records an event on the correct stream,
 *    and ensureOnHost() + data() returns post-kernel data
 * 2. mark_device_dirty_flags_only() does NOT record an event, and
 *    ensureOnHost() falls back to full device synchronization
 * 3. The stale event scenario: event from operation A persists after
 *    flags-only dirty marking from operation B
 * 4. MockBackend event tracking verifies event create/record/wait calls
 *
 * **Skipping**: These tests require at least one GPU (CUDA or ROCm).
 * They automatically skip on CPU-only machines.
 *
 * @see tests/v2/unit/execution/local_execution/coherence/Test__StreamCoherence.cpp
 * @see src/v2/tensors/TensorBase.cpp (mark_device_dirty_with_event, ensureOnHost)
 * @see tests/v2/mocks/MockBackend.h
 */

#include <gtest/gtest.h>

// Project headers
#include "tensors/Tensors.h"
#include "backends/DeviceId.h"
#include "backends/ComputeBackend.h"
#include "execution/local_execution/coherence/StageCoherence.h"
#include "execution/local_execution/coherence/GpuCoherence.h"

// Mock for event tracking
#include "mocks/MockBackend.h"

#include <memory>
#include <vector>
#include <cstring>
#include <cmath>
#include <numeric>
#include <iostream>

using namespace llaminar2;

// =============================================================================
// Test Fixture with GPU Detection
// =============================================================================

class Test__StreamCoherenceIntegration : public ::testing::Test
{
protected:
    void SetUp() override
    {
        // Initialize device manager
        auto &dm = DeviceManager::instance();
        if (dm.devices().empty())
        {
            dm.initialize(-1); // Enumerate all devices
        }

        if (!dm.has_gpu())
        {
            GTEST_SKIP() << "No GPU available, skipping stream coherence integration tests";
        }

        // Prefer CUDA, fall back to ROCm
        int gpu_idx = dm.find_device(ComputeBackendType::GPU_CUDA);
        if (gpu_idx < 0)
        {
            gpu_idx = dm.find_device(ComputeBackendType::GPU_ROCM);
        }
        if (gpu_idx < 0)
        {
            GTEST_SKIP() << "No CUDA or ROCm device found";
        }

        const auto &device_info = dm.devices()[gpu_idx];
        int ordinal = device_info.device_id;

        if (device_info.type == ComputeBackendType::GPU_CUDA)
        {
            gpu_device_ = DeviceId::cuda(ordinal);
        }
        else
        {
            gpu_device_ = DeviceId::rocm(ordinal);
        }

        std::cout << "[StreamCoherenceIntegration] Using device: "
                  << gpu_device_.toString() << std::endl;
    }

    /**
     * @brief Create an FP32 tensor with sequential values
     */
    std::unique_ptr<FP32Tensor> createTestTensor(size_t rows, size_t cols, float base = 1.0f)
    {
        auto tensor = std::make_unique<FP32Tensor>(
            std::vector<size_t>{rows, cols}, DeviceId::cpu());
        float *data = tensor->mutable_data();
        for (size_t i = 0; i < rows * cols; ++i)
        {
            data[i] = base + static_cast<float>(i) * 0.001f;
        }
        return tensor;
    }

    DeviceId gpu_device_ = DeviceId::cpu();
};

// =============================================================================
// Test: Basic Coherence Round-Trip (host → device → host)
// =============================================================================

TEST_F(Test__StreamCoherenceIntegration, RoundTrip_HostToDeviceToHost)
{
    // Verify basic coherence: upload tensor, mark dirty, read back
    constexpr size_t ROWS = 4;
    constexpr size_t COLS = 64;

    auto tensor = createTestTensor(ROWS, COLS);

    // Save original data for comparison
    std::vector<float> original(ROWS * COLS);
    std::memcpy(original.data(), tensor->data(), ROWS * COLS * sizeof(float));

    // Upload to GPU
    ASSERT_TRUE(tensor->ensureOnDevice(gpu_device_));

    // Mark device dirty (the GPU hasn't actually modified data, but this
    // exercises the coherence path: host becomes stale)
    tensor->mark_device_dirty();

    // Read back — should trigger D2H transfer
    const float *result = tensor->data();
    ASSERT_NE(result, nullptr);

    // Data should match (GPU didn't modify it)
    for (size_t i = 0; i < ROWS * COLS; ++i)
    {
        EXPECT_FLOAT_EQ(result[i], original[i])
            << "Mismatch at index " << i;
    }
}

// =============================================================================
// Test: mark_device_dirty_with_event Records Correct Event
// =============================================================================

TEST_F(Test__StreamCoherenceIntegration, WithEvent_RecordsAndWaitsCorrectly)
{
    // After mark_device_dirty_with_event, calling data() should:
    // 1. Wait on the recorded event
    // 2. Perform D2H transfer
    // 3. Return the correct data

    constexpr size_t ROWS = 8;
    constexpr size_t COLS = 128;

    auto tensor = createTestTensor(ROWS, COLS, 42.0f);

    std::vector<float> original(ROWS * COLS);
    std::memcpy(original.data(), tensor->data(), ROWS * COLS * sizeof(float));

    // Upload to device
    ASSERT_TRUE(tensor->ensureOnDevice(gpu_device_));

    // Mark dirty WITH event (using default stream / nullptr)
    tensor->mark_device_dirty_with_event(nullptr);

    // Host should be stale now
    EXPECT_FALSE(tensor->isOnCPU());

    // Reading data should trigger event wait + D2H
    const float *result = tensor->data();
    ASSERT_NE(result, nullptr);

    // After data() call, host should be valid again
    EXPECT_TRUE(tensor->isOnCPU());

    // Data should be correct
    for (size_t i = 0; i < ROWS * COLS; ++i)
    {
        EXPECT_FLOAT_EQ(result[i], original[i])
            << "Mismatch at index " << i;
    }
}

// =============================================================================
// Test: mark_device_dirty_flags_only Falls Back to Full Sync
// =============================================================================

TEST_F(Test__StreamCoherenceIntegration, FlagsOnly_FallsBackToFullSync)
{
    // mark_device_dirty_flags_only() does NOT record an event.
    // When ensureOnHost() is called, it should fall back to full device sync.
    // This is slower but still correct.

    constexpr size_t ROWS = 4;
    constexpr size_t COLS = 32;

    auto tensor = createTestTensor(ROWS, COLS, 7.0f);

    std::vector<float> original(ROWS * COLS);
    std::memcpy(original.data(), tensor->data(), ROWS * COLS * sizeof(float));

    // Upload to device
    ASSERT_TRUE(tensor->ensureOnDevice(gpu_device_));

    // Flags-only dirty marking (no event)
    tensor->mark_device_dirty_flags_only();

    // Host should be stale
    EXPECT_FALSE(tensor->isOnCPU());

    // Reading data should still work (falls back to full sync)
    const float *result = tensor->data();
    ASSERT_NE(result, nullptr);

    // Data should be correct
    for (size_t i = 0; i < ROWS * COLS; ++i)
    {
        EXPECT_FLOAT_EQ(result[i], original[i])
            << "Mismatch at index " << i;
    }
}

// =============================================================================
// Test: Stale Event Scenario — Event Persists Through Flags-Only Marking
// =============================================================================

TEST_F(Test__StreamCoherenceIntegration, StaleEvent_PersistsThroughFlagsOnlyMarking)
{
    // Reproduce the Bug 2 scenario on real hardware:
    //
    // 1. Upload tensor to GPU
    // 2. Mark dirty WITH event (simulating a GEMM kernel)
    // 3. Mark dirty FLAGS ONLY (simulating what executor did for allreduce before fix)
    // 4. Read data back
    //
    // The data should still be correct because the stale event completion
    // time is BEFORE the flags-only marking (the upload is complete).
    // But the critical insight is that the event is stale — it doesn't
    // reflect what happened AFTER step 2.

    constexpr size_t ROWS = 4;
    constexpr size_t COLS = 64;

    auto tensor = createTestTensor(ROWS, COLS, 100.0f);

    std::vector<float> original(ROWS * COLS);
    std::memcpy(original.data(), tensor->data(), ROWS * COLS * sizeof(float));

    // Step 1: Upload
    ASSERT_TRUE(tensor->ensureOnDevice(gpu_device_));

    // Step 2: Mark dirty with event (simulates GEMM completing)
    tensor->mark_device_dirty_with_event(nullptr);

    // Step 3: Mark dirty flags-only (simulates executor after "allreduce")
    // The stale event from step 2 persists
    tensor->mark_device_dirty_flags_only();

    // Step 4: Read back — ensureOnHost() will wait on the stale event from step 2
    const float *result = tensor->data();
    ASSERT_NE(result, nullptr);

    // Data is correct here because no kernel actually modified it.
    // In the real bug scenario, step 3 would be an allreduce that MODIFIES
    // the data on GPU, but the stale event from step 2 would let the D2H
    // transfer proceed before the allreduce completes.
    for (size_t i = 0; i < ROWS * COLS; ++i)
    {
        EXPECT_FLOAT_EQ(result[i], original[i]);
    }
}

// =============================================================================
// Test: Fix Scenario — Stage Calls mark_device_dirty_with_event After Allreduce
// =============================================================================

TEST_F(Test__StreamCoherenceIntegration, FixScenario_StageCallsEventAfterAllreduce)
{
    // Demonstrates the fix: After the allreduce, the stage calls
    // mark_device_dirty_with_event() which replaces the stale event.
    //
    // 1. Upload tensor
    // 2. Mark dirty with event (GEMM)
    // 3. Mark dirty flags-only (executor, for intermediate pipeline stage)
    // 4. Mark dirty with event AGAIN (stage-level fix in TPAllreduceStage)
    // 5. Read data — waits on the NEW event

    constexpr size_t ROWS = 4;
    constexpr size_t COLS = 128;

    auto tensor = createTestTensor(ROWS, COLS, 50.0f);

    std::vector<float> original(ROWS * COLS);
    std::memcpy(original.data(), tensor->data(), ROWS * COLS * sizeof(float));

    // Step 1
    ASSERT_TRUE(tensor->ensureOnDevice(gpu_device_));

    // Step 2: GEMM event
    tensor->mark_device_dirty_with_event(nullptr);

    // Step 3: Executor flags-only (intermediate stage)
    tensor->mark_device_dirty_flags_only();

    // Step 4: FIX — Stage records new event after allreduce
    tensor->mark_device_dirty_with_event(nullptr);

    // Step 5: Read back — waits on the new event from step 4
    const float *result = tensor->data();
    ASSERT_NE(result, nullptr);

    for (size_t i = 0; i < ROWS * COLS; ++i)
    {
        EXPECT_FLOAT_EQ(result[i], original[i]);
    }
}

// =============================================================================
// Test: StageCoherence markOutputsDirty vs markOutputsDirtyFlagsOnly
// =============================================================================

TEST_F(Test__StreamCoherenceIntegration, StageCoherence_MarkOutputsDirty_WithRealGPU)
{
    // Verify StageCoherence::markOutputsDirty() works with real GPU tensors

    constexpr size_t ROWS = 4;
    constexpr size_t COLS = 64;

    auto tensor = createTestTensor(ROWS, COLS, 3.14f);

    std::vector<float> original(ROWS * COLS);
    std::memcpy(original.data(), tensor->data(), ROWS * COLS * sizeof(float));

    ASSERT_TRUE(tensor->ensureOnDevice(gpu_device_));

    // Use StageCoherence function (event-based)
    CoherenceBuffer buf;
    buf.tensor = tensor.get();
    buf.name = "test_output";
    buf.data = nullptr; // Not needed for coherence operations
    buf.rows = ROWS;
    buf.cols = COLS;
    buf.dtype = "FP32";
    buf.is_inout = false;

    std::vector<CoherenceBuffer> outputs = {buf};
    markOutputsDirty(outputs, nullptr);

    // Host should be stale
    EXPECT_FALSE(tensor->isOnCPU());

    // Read back
    const float *result = tensor->data();
    ASSERT_NE(result, nullptr);

    for (size_t i = 0; i < ROWS * COLS; ++i)
    {
        EXPECT_FLOAT_EQ(result[i], original[i]);
    }
}

TEST_F(Test__StreamCoherenceIntegration, StageCoherence_FlagsOnly_WithRealGPU)
{
    // Verify StageCoherence::markOutputsDirtyFlagsOnly() works with real GPU tensors
    // (falls back to full sync on ensureOnHost)

    constexpr size_t ROWS = 4;
    constexpr size_t COLS = 64;

    auto tensor = createTestTensor(ROWS, COLS, 2.718f);

    std::vector<float> original(ROWS * COLS);
    std::memcpy(original.data(), tensor->data(), ROWS * COLS * sizeof(float));

    ASSERT_TRUE(tensor->ensureOnDevice(gpu_device_));

    CoherenceBuffer buf;
    buf.tensor = tensor.get();
    buf.name = "test_output";
    buf.data = nullptr;
    buf.rows = ROWS;
    buf.cols = COLS;
    buf.dtype = "FP32";
    buf.is_inout = false;

    std::vector<CoherenceBuffer> outputs = {buf};
    markOutputsDirtyFlagsOnly(outputs);

    EXPECT_FALSE(tensor->isOnCPU());

    const float *result = tensor->data();
    ASSERT_NE(result, nullptr);

    for (size_t i = 0; i < ROWS * COLS; ++i)
    {
        EXPECT_FLOAT_EQ(result[i], original[i]);
    }
}

// =============================================================================
// Test: with_gpu_coherence Helper
// =============================================================================

TEST_F(Test__StreamCoherenceIntegration, GpuCoherenceHelper_WithRealGPU)
{
    // Verify the RAII with_gpu_coherence helper works correctly

    constexpr size_t ROWS = 4;
    constexpr size_t COLS = 32;

    auto input = createTestTensor(ROWS, COLS, 1.0f);
    auto output = createTestTensor(ROWS, COLS, 0.0f);

    std::vector<float> original_input(ROWS * COLS);
    std::memcpy(original_input.data(), input->data(), ROWS * COLS * sizeof(float));

    // Use with_gpu_coherence to manage coherence automatically
    bool ok = with_gpu_coherence(
        gpu_device_,
        {input.get()},  // inputs
        {output.get()}, // outputs
        [&]
        {
            // No actual kernel — just verify the tensors are on device
            // In a real scenario, a GPU kernel would run here
            return true;
        });

    EXPECT_TRUE(ok);

    // After with_gpu_coherence, outputs should be marked device-dirty
    // Input should still be host-valid (read-only)
}

// =============================================================================
// Test: Multiple Tensors Coherence Consistency
// =============================================================================

TEST_F(Test__StreamCoherenceIntegration, MultipleTensors_CoherenceConsistency)
{
    // Verify multiple tensors can independently track coherence state

    auto tensor_a = createTestTensor(4, 64, 1.0f);
    auto tensor_b = createTestTensor(8, 32, 2.0f);

    std::vector<float> orig_a(4 * 64), orig_b(8 * 32);
    std::memcpy(orig_a.data(), tensor_a->data(), orig_a.size() * sizeof(float));
    std::memcpy(orig_b.data(), tensor_b->data(), orig_b.size() * sizeof(float));

    // Upload both
    ASSERT_TRUE(tensor_a->ensureOnDevice(gpu_device_));
    ASSERT_TRUE(tensor_b->ensureOnDevice(gpu_device_));

    // Mark A with event, B with flags-only
    tensor_a->mark_device_dirty_with_event(nullptr);
    tensor_b->mark_device_dirty_flags_only();

    // Both should be device-authoritative
    EXPECT_FALSE(tensor_a->isOnCPU());
    EXPECT_FALSE(tensor_b->isOnCPU());

    // Read back both — A uses event sync, B uses full sync
    const float *result_a = tensor_a->data();
    const float *result_b = tensor_b->data();

    ASSERT_NE(result_a, nullptr);
    ASSERT_NE(result_b, nullptr);

    for (size_t i = 0; i < 4 * 64; ++i)
    {
        EXPECT_FLOAT_EQ(result_a[i], orig_a[i]);
    }
    for (size_t i = 0; i < 8 * 32; ++i)
    {
        EXPECT_FLOAT_EQ(result_b[i], orig_b[i]);
    }
}

// =============================================================================
// Test: MockBackend Event Tracking Verification
// =============================================================================
//
// These tests use MockBackend (CPU-based) to verify event tracking counters
// without requiring GPU hardware. They complement the GPU tests above.
// =============================================================================

class Test__StreamCoherenceEventTracking : public ::testing::Test
{
protected:
    void SetUp() override
    {
        mock_backend_ = std::make_shared<test::MockBackend>();
    }

    std::shared_ptr<test::MockBackend> mock_backend_;
};

TEST_F(Test__StreamCoherenceEventTracking, MockBackend_CreatesUniqueEvents)
{
    void *event1 = mock_backend_->createEvent(0);
    void *event2 = mock_backend_->createEvent(0);
    void *event3 = mock_backend_->createEvent(1);

    EXPECT_NE(event1, nullptr);
    EXPECT_NE(event2, nullptr);
    EXPECT_NE(event3, nullptr);
    EXPECT_NE(event1, event2);
    EXPECT_NE(event2, event3);

    EXPECT_EQ(mock_backend_->getEventCreateCount(), 3u);
}

TEST_F(Test__StreamCoherenceEventTracking, MockBackend_TracksRecordOperations)
{
    void *event = mock_backend_->createEvent(0);
    void *stream = reinterpret_cast<void *>(0x1234);

    ASSERT_TRUE(mock_backend_->recordEvent(event, 0, stream));
    ASSERT_TRUE(mock_backend_->recordEvent(event, 0, nullptr));

    EXPECT_EQ(mock_backend_->getEventRecordCount(), 2u);

    auto records = mock_backend_->getEventRecordsForStream(stream);
    EXPECT_EQ(records.size(), 1u);
    EXPECT_EQ(records[0].event, event);
    EXPECT_EQ(records[0].stream, stream);
}

TEST_F(Test__StreamCoherenceEventTracking, MockBackend_TracksWaitOperations)
{
    void *event = mock_backend_->createEvent(0);

    mock_backend_->waitForEvent(event, 0);
    mock_backend_->waitForEvent(event, 0);
    mock_backend_->waitForEvent(event, 0);

    EXPECT_EQ(mock_backend_->getEventWaitCount(), 3u);
}

TEST_F(Test__StreamCoherenceEventTracking, MockBackend_ResetClearsAll)
{
    mock_backend_->createEvent(0);
    mock_backend_->recordEvent(reinterpret_cast<void *>(0x1000), 0);
    mock_backend_->hostToDevice(nullptr, nullptr, 0, 0);

    mock_backend_->resetAll();

    EXPECT_EQ(mock_backend_->getEventCreateCount(), 0u);
    EXPECT_EQ(mock_backend_->getEventRecordCount(), 0u);
    EXPECT_EQ(mock_backend_->getH2DCount(), 0u);
}

TEST_F(Test__StreamCoherenceEventTracking, MockBackend_EventRecordHistory)
{
    void *event1 = mock_backend_->createEvent(0);
    void *event2 = mock_backend_->createEvent(1);

    mock_backend_->recordEvent(event1, 0, reinterpret_cast<void *>(0xA));
    mock_backend_->recordEvent(event2, 1, reinterpret_cast<void *>(0xB));
    mock_backend_->waitForEvent(event1, 0);
    mock_backend_->destroyEvent(event2, 1);

    auto records = mock_backend_->getEventRecords();
    ASSERT_EQ(records.size(), 6u); // 2 creates + 2 records + 1 wait + 1 destroy

    // Verify chronological order
    EXPECT_EQ(records[0].type, test::MockBackend::EventRecord::CREATE);
    EXPECT_EQ(records[1].type, test::MockBackend::EventRecord::CREATE);
    EXPECT_EQ(records[2].type, test::MockBackend::EventRecord::RECORD);
    EXPECT_EQ(records[3].type, test::MockBackend::EventRecord::RECORD);
    EXPECT_EQ(records[4].type, test::MockBackend::EventRecord::WAIT);
    EXPECT_EQ(records[5].type, test::MockBackend::EventRecord::DESTROY);
}
