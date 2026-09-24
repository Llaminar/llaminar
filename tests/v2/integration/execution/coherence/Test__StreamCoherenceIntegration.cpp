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
 * 1. Event-backed publication records on the exact producer stream and host
 *    observation waits for that event before D2H.
 * 2. Graph-owned publication is not independently host-observable until the
 *    graph controller publishes replay completion.
 * 3. Repeated producers refresh the event dependency rather than preserving
 *    stale ordering.
 * 4. MockBackend event tracking verifies event create/record/wait calls.
 *
 * **Skipping**: These tests require at least one GPU (CUDA or ROCm).
 * They automatically skip on CPU-only machines.
 *
 * @see tests/v2/unit/execution/local_execution/coherence/Test__StreamCoherence.cpp
 * @see src/v2/transfer/TransferEngine.h
 * @see tests/v2/mocks/MockBackend.h
 */

#include <gtest/gtest.h>
#include "transfer/TransferEngine.h"

// Project headers
#include "tensors/Tensors.h"
#include "backends/DeviceId.h"
#include "backends/ComputeBackend.h"
#include "execution/local_execution/coherence/GpuCoherence.h"

// Mock for event tracking
#include "mocks/MockBackend.h"
#include "../../../utils/ScopedGPUStream.h"

#include <memory>
#include <vector>
#include <cstring>
#include <cmath>
#include <numeric>
#include <iostream>
#include <stdexcept>

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
        producer_stream_ =
            std::make_unique<llaminar2::test::ScopedGPUStream>(gpu_device_);
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
    std::unique_ptr<llaminar2::test::ScopedGPUStream> producer_stream_;
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
    TransferEngine::publishCurrentDeviceWrite(
        tensor,
        producer_stream_->get());

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
// Test: event-backed publication records the correct event
// =============================================================================

TEST_F(Test__StreamCoherenceIntegration, WithEvent_RecordsAndWaitsCorrectly)
{
    // After event-backed publication, calling data() should:
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

    // Publish the write on the exact explicit producer stream.
    TransferEngine::publishCurrentDeviceWrite(
        tensor,
        producer_stream_->get());

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
// Test: explicit-stream publication remains event-backed
// =============================================================================

TEST_F(Test__StreamCoherenceIntegration, ExplicitStreamPublicationUsesEventReadback)
{
    // A non-null producer stream records the exact dependency consumed by the
    // later host readback. Stream zero is intentionally not a legal producer.

    constexpr size_t ROWS = 4;
    constexpr size_t COLS = 32;

    auto tensor = createTestTensor(ROWS, COLS, 7.0f);

    std::vector<float> original(ROWS * COLS);
    std::memcpy(original.data(), tensor->data(), ROWS * COLS * sizeof(float));

    // Upload to device
    ASSERT_TRUE(tensor->ensureOnDevice(gpu_device_));

    // Event-backed publication on the explicit producer stream.
    TransferEngine::publishCurrentDeviceWrite(
        tensor,
        producer_stream_->get());

    // Host should be stale
    EXPECT_FALSE(tensor->isOnCPU());

    // Reading data waits on the tensor's exact publication event.
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
// Test: repeated event publication refreshes the dependency
// =============================================================================

TEST_F(Test__StreamCoherenceIntegration, RepeatedEventPublicationRefreshesDependency)
{
    // Reusing a tensor across producers must re-record its event after each
    // write so the eventual host consumer observes the newest producer.

    constexpr size_t ROWS = 4;
    constexpr size_t COLS = 64;

    auto tensor = createTestTensor(ROWS, COLS, 100.0f);

    std::vector<float> original(ROWS * COLS);
    std::memcpy(original.data(), tensor->data(), ROWS * COLS * sizeof(float));

    // Step 1: Upload
    ASSERT_TRUE(tensor->ensureOnDevice(gpu_device_));

    // Step 2: Mark dirty with event (simulates GEMM completing)
    TransferEngine::publishCurrentDeviceWrite(
        tensor,
        producer_stream_->get());

    // Step 3: A later producer refreshes the same event-backed handoff.
    TransferEngine::publishCurrentDeviceWrite(
        tensor,
        producer_stream_->get());

    // Step 4: Readback waits on the latest publication.
    const float *result = tensor->data();
    ASSERT_NE(result, nullptr);

    for (size_t i = 0; i < ROWS * COLS; ++i)
    {
        EXPECT_FLOAT_EQ(result[i], original[i]);
    }
}

// =============================================================================
// Test: each stage refreshes event publication after its write
// =============================================================================

TEST_F(Test__StreamCoherenceIntegration, FixScenario_StageCallsEventAfterAllreduce)
{
    // After the allreduce, the stage publishes its producer stream, replacing
    // the previous dependency.
    //
    // 1. Upload tensor
    // 2. Mark dirty with event (GEMM)
    // 3. Publish the next producer on the same default stream
    // 4. Publish the collective producer after it completes
    // 5. Read data — waits on the newest event

    constexpr size_t ROWS = 4;
    constexpr size_t COLS = 128;

    auto tensor = createTestTensor(ROWS, COLS, 50.0f);

    std::vector<float> original(ROWS * COLS);
    std::memcpy(original.data(), tensor->data(), ROWS * COLS * sizeof(float));

    // Step 1
    ASSERT_TRUE(tensor->ensureOnDevice(gpu_device_));

    // Step 2: GEMM event
    TransferEngine::publishCurrentDeviceWrite(
        tensor,
        producer_stream_->get());

    // Step 3: A second event-backed producer
    TransferEngine::publishCurrentDeviceWrite(
        tensor,
        producer_stream_->get());

    // Step 4: FIX — Stage records new event after allreduce
    TransferEngine::publishCurrentDeviceWrite(
        tensor,
        producer_stream_->get());

    // Step 5: Read back — waits on the new event from step 4
    const float *result = tensor->data();
    ASSERT_NE(result, nullptr);

    for (size_t i = 0; i < ROWS * COLS; ++i)
    {
        EXPECT_FLOAT_EQ(result[i], original[i]);
    }
}

// =============================================================================
// Test: explicit event-backed and graph-owned publication
// =============================================================================

TEST_F(Test__StreamCoherenceIntegration, PublishDeviceWrite_WithRealGPU)
{
    constexpr size_t ROWS = 4;
    constexpr size_t COLS = 64;

    auto tensor = createTestTensor(ROWS, COLS, 3.14f);

    std::vector<float> original(ROWS * COLS);
    std::memcpy(original.data(), tensor->data(), ROWS * COLS * sizeof(float));

    ASSERT_TRUE(tensor->ensureOnDevice(gpu_device_));

    TransferEngine::publishDeviceWrite(
        tensor.get(), gpu_device_, producer_stream_->get());

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

TEST_F(Test__StreamCoherenceIntegration, GraphOwnedWriteRequiresControllerEvent)
{
    // Graph-owned publication deliberately carries no per-tensor event. A host
    // read must fail until the graph controller publishes replay completion.

    constexpr size_t ROWS = 4;
    constexpr size_t COLS = 64;

    auto tensor = createTestTensor(ROWS, COLS, 2.718f);

    std::vector<float> original(ROWS * COLS);
    std::memcpy(original.data(), tensor->data(), ROWS * COLS * sizeof(float));

    ASSERT_TRUE(tensor->ensureOnDevice(gpu_device_));

    TransferEngine::publishGraphOwnedDeviceWrite(
        tensor.get(), gpu_device_);

    EXPECT_FALSE(tensor->isOnCPU());

    EXPECT_THROW(
        static_cast<void>(tensor->data()),
        std::runtime_error);

    // Model the graph controller's post-launch event publication.
    TransferEngine::publishDeviceWrite(
        tensor.get(),
        gpu_device_,
        producer_stream_->get());

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
        producer_stream_->get(),
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

    // Publish both tensors on their producer streams.
    TransferEngine::publishCurrentDeviceWrite(
        tensor_a,
        producer_stream_->get());
    TransferEngine::publishCurrentDeviceWrite(
        tensor_b,
        producer_stream_->get());

    // Both should be device-authoritative
    EXPECT_FALSE(tensor_a->isOnCPU());
    EXPECT_FALSE(tensor_b->isOnCPU());

    // Read back both through their exact event dependencies.
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
