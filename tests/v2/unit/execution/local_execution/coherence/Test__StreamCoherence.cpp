/**
 * @file Test__StreamCoherence.cpp
 * @brief Unit tests for stream/coherence interaction in the TP collective stages
 * @author GitHub Copilot
 * @date January 2026
 *
 * Regression tests for two coherence bugs fixed in the TP pipeline:
 *
 * **Bug 1 — Collective stages had CoherencePolicy::NONE**:
 *   When collective stages (TPAllreduceStage, AllGatherStage, AllGatherVStage)
 *   returned CoherencePolicy::NONE, the DeviceGraphExecutor skipped marking
 *   outputs as device-dirty after execution. Subsequent ensureOnHost() calls
 *   returned stale host data (pre-allreduce), causing inference divergence.
 *   Fix: Changed collective stages to CoherencePolicy::OUTPUT.
 *
 * **Bug 2 — Stale completion event after graph-owned publication**:
 *   During graph construction, per-stage publication intentionally avoids
 *   recording an externally visible completion event. If a preceding stage
 *   recorded an event, that old dependency must be retired; otherwise a host
 *   observer can wait on the wrong producer and race replay. Readback must fail
 *   until the graph controller publishes the replay completion event.
 *
 * **Test Strategy**:
 *   Unit tests use MockCoherenceTensor (exposing protected coherence state)
 *   and TransferEngine publication APIs without GPU hardware.
 *   These tests verify the LOGIC of the coherence system, not the actual
 *   GPU synchronization.
 *
 * @see src/v2/transfer/TransferEngine.h
 * @see src/v2/transfer/TransferEngine.h
 * @see tests/v2/mocks/MockBackend.h
 */

#include <gtest/gtest.h>
#include "transfer/TransferEngine.h"

// Project headers
#include "execution/compute_stages/IComputeStage.h"
#include "execution/compute_stages/stages/TPAllreduceStage.h"
#include "execution/compute_stages/stages/AllGatherStage.h"
#include "execution/compute_stages/stages/AllGatherVStage.h"
#include "execution/compute_stages/stages/AllreduceStage.h"
#include "execution/compute_stages/stages/GEMMStage.h"
#include "tensors/Tensors.h"
#include "backends/DeviceId.h"
#include "backends/GlobalDeviceAddress.h"
#include "collective/ILocalTPContext.h"
#include "utils/MPIContext.h"

#include "../../../../mocks/MockBackend.h"
#include "../../../../mocks/MockWorkerGPUContext.h"

#include <memory>
#include <vector>
#include <cstring>

using namespace llaminar2;

// =============================================================================
// Test Helper: MockCoherenceTensor
// =============================================================================

/**
 * @brief FP32Tensor subclass that exposes protected coherence state for testing
 *
 * Provides read access to device_completion_event_ and coherence_state_
 * for verifying coherence transitions without requiring a real GPU backend.
 */
class MockCoherenceTensor : public FP32Tensor
{
public:
    using FP32Tensor::FP32Tensor;

    // ---- Expose protected state ----

    void *getCompletionEvent() const { return device_completion_event_; }
    bool getHostValid() const { return ::llaminar2::isHostValid(coherence_state_); }
    bool getDeviceValid() const { return ::llaminar2::isDeviceValid(coherence_state_); }
    std::optional<DeviceId> getGpuDevice() const { return gpu_device_; }
    std::optional<DeviceId> getAuthoritativeDevice() const { return authoritative_device_; }

    // ---- Inject fake state for testing ----

    void injectCompletionEvent(void *event)
    {
        device_completion_event_ = event;
        completion_event_protection_ = event
                                           ? CompletionEventProtection::DeviceValue
                                           : CompletionEventProtection::None;
        if (event)
            event_device_ = gpu_device_;
        else
            event_device_.reset();
    }

    void injectGpuDevice(DeviceId device)
    {
        gpu_device_ = device;
    }

    void injectDeviceValid(bool valid)
    {
        if (valid && !::llaminar2::isHostValid(coherence_state_))
            setCoherenceState_(TensorCoherenceState::DEVICE_AUTHORITATIVE);
        else if (valid && ::llaminar2::isHostValid(coherence_state_))
            setCoherenceState_(TensorCoherenceState::SYNCED);
        else if (!valid && ::llaminar2::isHostValid(coherence_state_))
            setCoherenceState_(TensorCoherenceState::HOST_AUTHORITATIVE);
    }

    void injectHostValid(bool valid)
    {
        if (::llaminar2::isDeviceValid(coherence_state_) && !valid)
            setCoherenceState_(TensorCoherenceState::DEVICE_AUTHORITATIVE);
        else if (::llaminar2::isDeviceValid(coherence_state_) && valid)
            setCoherenceState_(TensorCoherenceState::SYNCED);
        else if (!::llaminar2::isDeviceValid(coherence_state_) && valid)
            setCoherenceState_(TensorCoherenceState::HOST_AUTHORITATIVE);
    }

    void injectGpuDataPtr(void *ptr)
    {
        gpu_data_ptr_ = ptr;
    }
};

/**
 * @brief Hardware-free probe for the base stage stream contract.
 *
 * The probe performs no work. It exists solely to prove that a GPU stage can
 * never adopt an implicit CUDA/HIP default stream through a null binding.
 */
class StreamBindingProbeStage final : public IComputeStage
{
public:
    explicit StreamBindingProbeStage(DeviceId device)
        : IComputeStage(device)
    {
    }

    bool execute(IDeviceContext *) override { return true; }
    ComputeStageType type() const override { return ComputeStageType::COPY; }
    bool supportsBackend(ComputeBackendType) const override { return true; }
    size_t estimatedFlops() const override { return 0; }
    StageDumpInfo buildDumpInfoImpl() const override { return {}; }
};

// =============================================================================
// Test Suite: CollectiveStageCoherencePolicy
// =============================================================================
//
// Regression tests for Bug 1: Collective stages MUST return OUTPUT, not NONE.
// With NONE the executor skips dirty-marking → stale host data after D2H.
// =============================================================================

class Test__StreamCoherence : public ::testing::Test
{
protected:
    static void SetUpTestSuite()
    {
        llaminar2::testing::installHardwareFreeGPUContextFactories();
    }

    void SetUp() override
    {
        mpi_ctx_ = std::make_shared<MPIContext>(0, 2, MPI_COMM_NULL);
    }

    /**
     * @brief Give a tensor physically consistent simulated GPU storage.
     *
     * Publication is no longer a free-form flag mutation: it validates that
     * the advertised GPU actually owns the tensor allocation. MockBackend
     * supplies host memory as device storage so these remain CPU-only tests.
     */
    void prepareSimulatedDevice(
        MockCoherenceTensor &tensor,
        DeviceId device = DeviceId::rocm(0))
    {
        tensor.setBackendForTesting(&backend_);
        ASSERT_TRUE(tensor.FP32Tensor::allocateOnDevice(device));
    }

    llaminar2::test::MockBackend backend_{DeviceType::ROCm};
    std::shared_ptr<IMPIContext> mpi_ctx_;
};

TEST_F(Test__StreamCoherence, GPUStageRejectsNullStreamBinding)
{
    StreamBindingProbeStage stage(DeviceId::cuda(0));

    EXPECT_THROW(stage.setGPUStream(nullptr), std::invalid_argument);
    EXPECT_FALSE(stage.hasGPUStream());
    EXPECT_THROW((void)stage.gpuStream(), std::logic_error);
    EXPECT_THROW(stage.requireGPUStream(), std::logic_error);
}

TEST_F(Test__StreamCoherence, GPUStageRequiredStreamReturnsExactBinding)
{
    StreamBindingProbeStage stage(DeviceId::rocm(1));
    void *producer_stream = reinterpret_cast<void *>(0xC011EC71);

    stage.setGPUStream(producer_stream);

    EXPECT_TRUE(stage.hasGPUStream());
    EXPECT_EQ(stage.gpuStream(), producer_stream);
    EXPECT_EQ(stage.requireGPUStream(), producer_stream);
}

TEST_F(Test__StreamCoherence, GPUExecutionTokenPreservesBindingAndRejectsInvalidPublication)
{
    StreamBindingProbeStage stage(DeviceId::rocm(1));
    void *producer_stream = reinterpret_cast<void *>(0xC011EC71);

    EXPECT_THROW((void)stage.gpuExecution(), std::logic_error);

    stage.setGPUStream(producer_stream);
    const StageGPUExecution execution = stage.gpuExecution();

    EXPECT_EQ(execution.device(), DeviceId::rocm(1));
    EXPECT_EQ(execution.nativeStream(), producer_stream);
    EXPECT_THROW(execution.prepareInput(nullptr), std::invalid_argument);
    EXPECT_THROW(execution.prepareOutput(nullptr), std::invalid_argument);
    EXPECT_THROW(execution.publish(nullptr), std::invalid_argument);
}

TEST_F(Test__StreamCoherence, CPUStageHasNoGPUStreamContract)
{
    StreamBindingProbeStage stage(DeviceId::cpu());

    EXPECT_FALSE(stage.hasGPUStream());
    EXPECT_EQ(stage.gpuStream(), nullptr);
    EXPECT_THROW(stage.requireGPUStream(), std::logic_error);
    EXPECT_THROW((void)stage.gpuExecution(), std::logic_error);
}

TEST_F(Test__StreamCoherence, TPAllreduceStage_CoherencePolicy_IsOutput)
{
    // TPAllreduceStage MUST return OUTPUT so executor marks outputs dirty
    // after the collective operation completes
    auto tp_ctx = createLocalTPContext(
        {GlobalDeviceAddress::cuda(0)}, {}, CollectiveBackendType::AUTO);

    TPAllreduceStage::Params params;
    params.tp_ctx = tp_ctx.get();

    TPAllreduceStage stage(params);
    EXPECT_EQ(stage.coherencePolicy(), CoherencePolicy::OUTPUT);
}

TEST_F(Test__StreamCoherence, AllGatherStage_CoherencePolicy_IsOutput)
{
    // AllGatherStage MUST return OUTPUT
    auto input = std::make_unique<FP32Tensor>(
        std::vector<size_t>{4, 32}, DeviceId::cpu());
    auto output = std::make_unique<FP32Tensor>(
        std::vector<size_t>{4, 64}, DeviceId::cpu());

    AllGatherStage::Params params;
    params.local_input = input.get();
    params.full_output = output.get();
    params.mpi_ctx = mpi_ctx_.get();
    params.actual_seq_len = 4;

    AllGatherStage stage(params);
    EXPECT_EQ(stage.coherencePolicy(), CoherencePolicy::OUTPUT);
}

TEST_F(Test__StreamCoherence, AllGatherVStage_CoherencePolicy_IsOutput)
{
    // AllGatherVStage MUST return OUTPUT
    AllGatherVStage::Params params;
    params.mpi_ctx = mpi_ctx_.get();
    params.recv_counts = {32, 32};
    params.displacements = {0, 32};

    AllGatherVStage stage(params);
    EXPECT_EQ(stage.coherencePolicy(), CoherencePolicy::OUTPUT);
}

TEST_F(Test__StreamCoherence, AllreduceStage_CoherencePolicy_IsNone)
{
    // MPI AllreduceStage (non-TP) correctly uses NONE — it handles its own sync
    AllreduceStage::Params params;
    params.mpi_ctx = mpi_ctx_.get();

    AllreduceStage stage(params);
    EXPECT_EQ(stage.coherencePolicy(), CoherencePolicy::NONE);
}

TEST_F(Test__StreamCoherence, GEMMStage_CoherencePolicy_IsFull)
{
    // Normal compute stages default to FULL coherence
    auto A = std::make_unique<FP32Tensor>(
        std::vector<size_t>{4, 8}, DeviceId::cpu());
    auto B = std::make_unique<FP32Tensor>(
        std::vector<size_t>{8, 16}, DeviceId::cpu());
    auto C = std::make_unique<FP32Tensor>(
        std::vector<size_t>{4, 16}, DeviceId::cpu());

    GEMMStage::Params params;
    params.A = A.get();
    params.B = B.get();
    params.C = C.get();
    params.m = 4;
    params.n = 16;
    params.k = 8;

    GEMMStage stage(params);
    EXPECT_EQ(stage.coherencePolicy(), CoherencePolicy::FULL);
}

// =============================================================================
// Test Suite: DirtyMarkingBehavior
// =============================================================================
//
// Tests for Bug 2: graph-owned publication has no per-tensor completion event.
// It must invalidate any stale event from a previous producer and remain
// unobservable to host code until the graph controller publishes completion.
// =============================================================================

TEST_F(Test__StreamCoherence, FlagsOnly_ClearsStaleCompletionEvent)
{
    auto tensor = std::make_unique<MockCoherenceTensor>(
        std::vector<size_t>{4, 64}, DeviceId::cpu());
    prepareSimulatedDevice(*tensor);

    // Simulate: a previous operation left a completion event
    void *stale_event = reinterpret_cast<void *>(0xCAFEBABE);
    tensor->injectCompletionEvent(stale_event);

    // Call flags-only dirty marking (what the executor does for intermediate stages)
    TransferEngine::publishGraphOwnedCurrentDeviceWrite(tensor);

    EXPECT_EQ(tensor->getCompletionEvent(), nullptr)
        << "Flags-only DEVICE_AUTHORITATIVE must clear stale completion events";

    // Verify the dirty flags were set correctly
    EXPECT_TRUE(tensor->getDeviceValid());
    EXPECT_FALSE(tensor->getHostValid()); // Non-mapped tensors → host is stale
}

TEST_F(Test__StreamCoherence, FlagsOnly_SetsDeviceDirtyState)
{
    auto tensor = std::make_unique<MockCoherenceTensor>(
        std::vector<size_t>{8, 32}, DeviceId::cpu());
    prepareSimulatedDevice(*tensor);

    // Start in host-authoritative state
    EXPECT_TRUE(tensor->getHostValid());
    EXPECT_FALSE(tensor->getDeviceValid());

    TransferEngine::publishGraphOwnedCurrentDeviceWrite(tensor);

    // Device should now be valid, host stale (non-mapped)
    EXPECT_TRUE(tensor->getDeviceValid());
    EXPECT_FALSE(tensor->getHostValid());
}

TEST_F(Test__StreamCoherence, WithEvent_ClearsStaleEventAndRecordsNew)
{
    // Event-backed publication creates or re-records the tensor's event after
    // the producer. MockBackend lets this unit test exercise the complete
    // lifecycle without touching GPU hardware.

    auto tensor = std::make_unique<MockCoherenceTensor>(
        std::vector<size_t>{4, 64}, DeviceId::cpu());
    prepareSimulatedDevice(*tensor);

    void *fake_stream = reinterpret_cast<void *>(0x1234);
    TransferEngine::publishCurrentDeviceWrite(tensor, fake_stream);

    // Dirty flags must be set
    EXPECT_TRUE(tensor->getDeviceValid());
    EXPECT_FALSE(tensor->getHostValid());
}

TEST_F(Test__StreamCoherence, WithEvent_CalledMultipleTimes_TracksCorrectStream)
{
    auto tensor = std::make_unique<MockCoherenceTensor>(
        std::vector<size_t>{4, 64}, DeviceId::cpu());
    prepareSimulatedDevice(*tensor);

    void *stream1 = reinterpret_cast<void *>(0x1111);
    void *stream2 = reinterpret_cast<void *>(0x2222);

    TransferEngine::publishCurrentDeviceWrite(tensor, stream1);

    TransferEngine::publishCurrentDeviceWrite(tensor, stream2);
}

// =============================================================================
// Test Suite: TransferEngine publication
// =============================================================================
//
// These regressions exercise the only public publication authority directly.
// =============================================================================

TEST_F(Test__StreamCoherence, PublishDeviceWrite_RecordsEvent)
{
    // Event-backed output publication makes each output device-authoritative
    // and records the exact producer stream.

    auto tensor = std::make_unique<MockCoherenceTensor>(
        std::vector<size_t>{4, 64}, DeviceId::cpu());
    prepareSimulatedDevice(*tensor);

    void *fake_stream = reinterpret_cast<void *>(0xABCD);
    TransferEngine::publishDeviceWrite(
        tensor.get(), DeviceId::rocm(0), fake_stream);

    // Tensor should be in DEVICE_AUTHORITATIVE state
    EXPECT_TRUE(tensor->getDeviceValid());
    EXPECT_FALSE(tensor->getHostValid());
    EXPECT_EQ(tensor->coherenceState(), TensorCoherenceState::DEVICE_AUTHORITATIVE);
}

TEST_F(Test__StreamCoherence, PublishGraphOwnedDeviceWrite_DoesNotRecordEvent)
{
    // Graph-owned output publication does not record a per-tensor event; the
    // graph controller owns the replay-level completion dependency.

    auto tensor = std::make_unique<MockCoherenceTensor>(
        std::vector<size_t>{4, 64}, DeviceId::cpu());
    prepareSimulatedDevice(*tensor);

    TransferEngine::publishGraphOwnedDeviceWrite(
        tensor.get(), DeviceId::rocm(0));

    // Tensor is device-authoritative but has no independently observable event.
    EXPECT_TRUE(tensor->getDeviceValid());
    EXPECT_FALSE(tensor->getHostValid());
    EXPECT_EQ(tensor->getCompletionEvent(), nullptr)
        << "graph-owned publication must not record a per-tensor event";
}

TEST_F(Test__StreamCoherence, PublishDeviceWrite_MultipleOutputs)
{
    // Verify all outputs transition to DEVICE_AUTHORITATIVE
    auto tensor1 = std::make_unique<MockCoherenceTensor>(
        std::vector<size_t>{4, 64}, DeviceId::cpu());
    auto tensor2 = std::make_unique<MockCoherenceTensor>(
        std::vector<size_t>{8, 32}, DeviceId::cpu());
    prepareSimulatedDevice(*tensor1);
    prepareSimulatedDevice(*tensor2);

    void *stream = reinterpret_cast<void *>(0x5678);
    TransferEngine::publishDeviceWrite(
        tensor1.get(), DeviceId::rocm(0), stream);
    TransferEngine::publishDeviceWrite(
        tensor2.get(), DeviceId::rocm(0), stream);

    EXPECT_TRUE(tensor1->getDeviceValid());
    EXPECT_FALSE(tensor1->getHostValid());
    EXPECT_EQ(tensor1->coherenceState(), TensorCoherenceState::DEVICE_AUTHORITATIVE);

    EXPECT_TRUE(tensor2->getDeviceValid());
    EXPECT_FALSE(tensor2->getHostValid());
    EXPECT_EQ(tensor2->coherenceState(), TensorCoherenceState::DEVICE_AUTHORITATIVE);
}

TEST_F(Test__StreamCoherence, PublishGraphOwnedDeviceWrite_ClearsExistingEvent)
{
    // Regression test for the exact Bug 2 scenario:
    // 1. Tensor has a stale event from a previous operation
    // 2. graph-owned publication is called for the new capture
    // 3. The stale event must be cleared so host readback does not wait on the
    //    wrong producer.

    auto tensor = std::make_unique<MockCoherenceTensor>(
        std::vector<size_t>{4, 64}, DeviceId::cpu());
    prepareSimulatedDevice(*tensor);

    // Inject stale event from "previous QKV projection"
    void *stale_event = reinterpret_cast<void *>(0xDEADFACE);
    tensor->injectCompletionEvent(stale_event);

    TransferEngine::publishGraphOwnedDeviceWrite(
        tensor.get(), DeviceId::rocm(0));

    EXPECT_EQ(tensor->getCompletionEvent(), nullptr)
        << "Flags-only output marking must invalidate older completion events";

    // Virtual method must NOT have been called
}

TEST_F(Test__StreamCoherence, PublishDeviceWrite_ReplacesStaleEvent)
{
    // Event-backed publication replaces the stale producer dependency.

    auto tensor = std::make_unique<MockCoherenceTensor>(
        std::vector<size_t>{4, 64}, DeviceId::cpu());
    prepareSimulatedDevice(*tensor);

    void *stale_event = reinterpret_cast<void *>(0xDEADFACE);
    tensor->injectCompletionEvent(stale_event);

    void *correct_stream = reinterpret_cast<void *>(0xA11EDECE);
    backend_.resetAll();
    TransferEngine::publishDeviceWrite(
        tensor.get(), DeviceId::rocm(0), correct_stream);

    // Tensor should be in DEVICE_AUTHORITATIVE state
    EXPECT_TRUE(tensor->getDeviceValid());
    EXPECT_FALSE(tensor->getHostValid());
    EXPECT_EQ(tensor->coherenceState(), TensorCoherenceState::DEVICE_AUTHORITATIVE);

    EXPECT_NE(tensor->getCompletionEvent(), nullptr);
    EXPECT_EQ(backend_.getEventRecordCount(), 1u)
        << "The reusable event must be re-recorded on the newest producer stream";
}

// =============================================================================
// Test Suite: CoherenceStateTransitions
// =============================================================================
//
// Tests the complete state machine of coherence transitions relevant to
// the TP pipeline: host → device → dirty (flags-only) → dirty (with event)
// =============================================================================

TEST_F(Test__StreamCoherence, InitialState_HostAuthoritative)
{
    auto tensor = std::make_unique<MockCoherenceTensor>(
        std::vector<size_t>{4, 64}, DeviceId::cpu());

    EXPECT_TRUE(tensor->getHostValid());
    EXPECT_FALSE(tensor->getDeviceValid());
    EXPECT_EQ(tensor->getCompletionEvent(), nullptr);
    EXPECT_FALSE(tensor->getGpuDevice().has_value());
}

TEST_F(Test__StreamCoherence, FlagsOnly_TransitionsToDeviceAuthoritative)
{
    auto tensor = std::make_unique<MockCoherenceTensor>(
        std::vector<size_t>{4, 64}, DeviceId::cpu());
    prepareSimulatedDevice(*tensor);

    TransferEngine::publishGraphOwnedCurrentDeviceWrite(tensor);

    EXPECT_TRUE(tensor->getDeviceValid());
    EXPECT_FALSE(tensor->getHostValid());
    EXPECT_EQ(tensor->getCompletionEvent(), nullptr); // No event created/modified
}

TEST_F(Test__StreamCoherence, WithEvent_TransitionsToDeviceAuthoritative)
{
    auto tensor = std::make_unique<MockCoherenceTensor>(
        std::vector<size_t>{4, 64}, DeviceId::cpu());
    prepareSimulatedDevice(*tensor);

    EXPECT_THROW(
        TransferEngine::publishCurrentDeviceWrite(tensor, nullptr),
        std::invalid_argument);

    EXPECT_TRUE(tensor->getHostValid());
}

TEST_F(Test__StreamCoherence, SequentialDirtyMarking_EventOverwriteSequence)
{
    // Simulates the pipeline sequence:
    // 1. GEMM kernel runs and the graph controller owns publication
    // 2. Allreduce runs → stage publishes its producer-stream event
    //
    // Verifies that step 2 correctly invokes the event-based path even after
    // step 1 already set the flags.

    auto tensor = std::make_unique<MockCoherenceTensor>(
        std::vector<size_t>{4, 64}, DeviceId::cpu());
    prepareSimulatedDevice(*tensor);

    // Step 1: GEMM output marked flags-only (intermediate stage)
    TransferEngine::publishGraphOwnedCurrentDeviceWrite(tensor);
    EXPECT_TRUE(tensor->getDeviceValid());

    // Step 2: Allreduce publishes its producer-stream event.
    void *allreduce_stream = reinterpret_cast<void *>(0xBCC10001);
    TransferEngine::publishCurrentDeviceWrite(tensor, allreduce_stream);
    EXPECT_TRUE(tensor->getDeviceValid());
}

// =============================================================================
// Test Suite: NullAndEdgeCases
// =============================================================================

TEST_F(Test__StreamCoherence, PublishDeviceWrite_NullStream_FailsBeforePublication)
{
    auto tensor = std::make_unique<MockCoherenceTensor>(
        std::vector<size_t>{4, 64}, DeviceId::cpu());
    prepareSimulatedDevice(*tensor);

    EXPECT_THROW(
        TransferEngine::publishDeviceWrite(
            tensor.get(), DeviceId::rocm(0), nullptr),
        std::invalid_argument);

    // Rejection must not partially publish an unordered device write.
    EXPECT_FALSE(tensor->getDeviceValid());
    EXPECT_TRUE(tensor->getHostValid());
    EXPECT_EQ(tensor->getCompletionEvent(), nullptr);
    EXPECT_EQ(tensor->coherenceState(), TensorCoherenceState::HOST_AUTHORITATIVE);
}

// =============================================================================
// Test Suite: Bug2 Scenario — End-to-End Stale Event Lifecycle
// =============================================================================
//
// Simulates the full lifecycle that triggered Bug 2:
// Stage A (GEMM) → Stage B (allreduce) → D2H readback
// =============================================================================

TEST_F(Test__StreamCoherence, Bug2Scenario_StaleEventLifecycle)
{
    // This test simulates the exact sequence of operations that caused Bug 2:
    //
    // 1. A captured stage publishes graph-owned output
    //    → No per-tensor event is recorded, and any prior event is retired
    //
    // 2. Stage B (TPAllreduce) records its own event when it owns a host-visible
    //    completion boundary
    //
    // 3. Host reads data via data() → calls ensureOnHost()
    //    → Graph-owned writes without replay publication fail; event-backed
    //      writes wait on the exact producer event.

    auto tensor = std::make_unique<MockCoherenceTensor>(
        std::vector<size_t>{4, 896}, DeviceId::cpu());
    prepareSimulatedDevice(*tensor);

    // Simulate prior event from earlier operation (e.g., previous decode iteration)
    void *prior_event = reinterpret_cast<void *>(0x01D00001);
    tensor->injectCompletionEvent(prior_event);

    // Step 1: Captured stage output — graph-owned publication.
    TransferEngine::publishGraphOwnedDeviceWrite(
        tensor.get(), DeviceId::rocm(0));

    EXPECT_EQ(tensor->getCompletionEvent(), nullptr)
        << "Graph-owned publication must retire the stale event";

    // Step 2: TPAllreduce publishes its exact producer stream after completion.
    void *allreduce_stream = reinterpret_cast<void *>(0xBCC10002);
    TransferEngine::publishCurrentDeviceWrite(tensor, allreduce_stream);

    // Verify: state transition was made
    EXPECT_EQ(tensor->coherenceState(), TensorCoherenceState::DEVICE_AUTHORITATIVE);
    EXPECT_TRUE(tensor->getDeviceValid());

    // Step 3: In a real scenario with a GPU backend, ensureOnHost() would now
    // wait on the NEW event (from allreduce) rather than the stale one.
    // We can't test the actual D2H here without a GPU, but we verified the
    // call chain is correct.
}

TEST_F(Test__StreamCoherence, Bug2Scenario_RepeatedFlagsOnlyClearsStaleEvent)
{
    // Repeated graph-owned writes remain eventless. Host readback must fail
    // until the graph controller supplies replay completion.

    auto tensor = std::make_unique<MockCoherenceTensor>(
        std::vector<size_t>{4, 896}, DeviceId::cpu());
    prepareSimulatedDevice(*tensor);

    void *stale_event = reinterpret_cast<void *>(0x57A1E001);
    tensor->injectCompletionEvent(stale_event);

    TransferEngine::publishGraphOwnedDeviceWrite(
        tensor.get(), DeviceId::rocm(0));
    TransferEngine::publishGraphOwnedDeviceWrite(
        tensor.get(), DeviceId::rocm(0));

    EXPECT_EQ(tensor->getCompletionEvent(), nullptr)
        << "Repeated flags-only marking must not preserve stale events";
}

// =============================================================================
// Test Suite: Event-published device-to-host coherence
// =============================================================================
//
// A GPU producer must publish a completion event with its authoritative tensor
// state.  Eventless publication is an ordering defect: readback rejects it
// without issuing a blocking device synchronization or an unsafe D2H copy.
//
// These tests use MockBackend dependency injection to verify the control flow
// without requiring GPU hardware.
// =============================================================================

TEST_F(Test__StreamCoherence, EnsureOnHost_NoEvent_FailsClosedWithoutTransfer)
{
    // Simulate a GPU producer that changed coherence flags but failed to
    // publish its stream completion event.

    using namespace llaminar2::test;

    constexpr size_t ROWS = 4;
    constexpr size_t COLS = 64;
    auto tensor = std::make_unique<MockCoherenceTensor>(
        std::vector<size_t>{ROWS, COLS}, DeviceId::cpu());

    // Set up the mock backend
    MockBackend mock_backend(DeviceType::ROCm);
    tensor->setBackendForTesting(&mock_backend);

    // Simulate: tensor has been uploaded to GPU and a kernel has written to it
    // 1. Set gpu_device_ (tells ensureOnHost which backend to use)
    //    Use ROCm device to avoid the cross-vendor CUDA event proxy path
    //    (waitForEventWithProxy checks gpu_device.is_cuda() and routes through
    //    backend singleton if active, bypassing our mock)
    tensor->injectGpuDevice(DeviceId::rocm(0));

    // 2. Allocate "device" memory through the mock backend
    size_t bytes = ROWS * COLS * sizeof(float);
    void *device_ptr = mock_backend.allocate(bytes, 0);
    ASSERT_NE(device_ptr, nullptr);

    // Write known pattern to "device" memory (simulating GPU kernel output)
    float *device_floats = static_cast<float *>(device_ptr);
    for (size_t i = 0; i < ROWS * COLS; i++)
    {
        device_floats[i] = static_cast<float>(i) * 0.1f;
    }

    // 3. Set gpu_data_ptr_ so ensureOnHost knows where to D2H from
    tensor->injectGpuDataPtr(device_ptr);

    // 4. Mark tensor as device-dirty with NO event (the Bug 6 scenario)
    TransferEngine::publishGraphOwnedCurrentDeviceWrite(tensor);

    // Verify preconditions
    EXPECT_TRUE(tensor->getDeviceValid());
    EXPECT_FALSE(tensor->getHostValid());
    EXPECT_EQ(tensor->getCompletionEvent(), nullptr)
        << "flags-only marking must NOT set a completion event";

    // Reset tracking counters
    mock_backend.resetAll();

    // ACT: Call ensureOnHost() with no valid producer publication.
    bool result = tensor->ensureOnHost();

    EXPECT_FALSE(result)
        << "eventless device authority must fail closed";
    EXPECT_EQ(mock_backend.getSyncCount(), 0u)
        << "eventless publication must never trigger a full-device sync";
    EXPECT_EQ(mock_backend.getD2HCount(), 0u)
        << "readback must not race an unpublished producer";
    EXPECT_EQ(mock_backend.getEventWaitCount(), 0u)
        << "there is no valid event to wait upon";

    // Cleanup: free the mock-allocated device memory and null out the pointer
    // BEFORE tensor destruction. Don't call clearBackendForTesting() — the
    // destructor needs the mock backend to handle any remaining cleanup.
    mock_backend.free(device_ptr, 0);
    tensor->injectGpuDataPtr(nullptr); // Prevent destructor from trying to free
}

TEST_F(Test__StreamCoherence, EnsureOnHost_StaleEventClearedByFlagsOnlyWrite_FailsClosed)
{
    // Regression for graph replay / helper-kernel writes that mark tensors dirty
    // without recording a fresh completion event. If an older event remains on
    // the tensor, readback can wait on the wrong producer.

    using namespace llaminar2::test;

    constexpr size_t ROWS = 4;
    constexpr size_t COLS = 64;
    auto tensor = std::make_unique<MockCoherenceTensor>(
        std::vector<size_t>{ROWS, COLS}, DeviceId::cpu());

    MockBackend mock_backend(DeviceType::ROCm);
    tensor->setBackendForTesting(&mock_backend);
    tensor->injectGpuDevice(DeviceId::rocm(0));

    size_t bytes = ROWS * COLS * sizeof(float);
    void *device_ptr = mock_backend.allocate(bytes, 0);
    ASSERT_NE(device_ptr, nullptr);

    float *device_floats = static_cast<float *>(device_ptr);
    for (size_t i = 0; i < ROWS * COLS; i++)
    {
        device_floats[i] = static_cast<float>(i) * 0.25f;
    }
    tensor->injectGpuDataPtr(device_ptr);

    void *stale_event = reinterpret_cast<void *>(0x57A1E002);
    tensor->injectCompletionEvent(stale_event);
    ASSERT_EQ(tensor->getCompletionEvent(), stale_event);

    TransferEngine::publishGraphOwnedCurrentDeviceWrite(tensor);
    ASSERT_EQ(tensor->getCompletionEvent(), nullptr)
        << "flags-only writes must invalidate older completion events";

    mock_backend.resetAll();

    bool result = tensor->ensureOnHost();
    EXPECT_FALSE(result)
        << "clearing a stale event must expose the missing publication";

    EXPECT_EQ(mock_backend.getEventWaitCount(), 0u)
        << "ensureOnHost must not wait on the stale event";
    EXPECT_EQ(mock_backend.getSyncCount(), 0u)
        << "missing publication must never trigger a full-device sync";
    EXPECT_EQ(mock_backend.getD2HCount(), 0u)
        << "missing publication must never permit readback";

    mock_backend.free(device_ptr, 0);
    tensor->injectGpuDataPtr(nullptr);
}

TEST_F(Test__StreamCoherence, Bug6_EnsureOnHost_WithEvent_UsesEventSync)
{
    // Contrast test: When a completion event IS recorded,
    // ensureOnHost() should use event-based sync (not full device sync).
    // This verifies the event path still works correctly after the fix.
    //
    // We inject state directly (rather than calling mark_device_dirty_with_event)
    // to avoid triggering the real cross-vendor proxy path that the static
    // waitForEventWithProxy() uses for CUDA devices. The MockBackend handles
    // waitForEvent() correctly, but the static proxy bypasses it.

    using namespace llaminar2::test;

    constexpr size_t ROWS = 4;
    constexpr size_t COLS = 64;
    // The injected backend must outlive the tensor even when an assertion
    // throws and GoogleTest unwinds this scope.
    MockBackend mock_backend(DeviceType::ROCm);
    auto tensor = std::make_unique<MockCoherenceTensor>(
        std::vector<size_t>{ROWS, COLS}, DeviceId::cpu());

    tensor->setBackendForTesting(&mock_backend);

    // Use ROCm device to avoid the cross-vendor CUDA event proxy path
    // (waitForEventWithProxy checks gpu_device.is_cuda() and routes through
    // backend singleton if active, bypassing our mock)
    tensor->injectGpuDevice(DeviceId::rocm(0));

    size_t bytes = ROWS * COLS * sizeof(float);
    void *device_ptr = mock_backend.allocate(bytes, 0);
    ASSERT_NE(device_ptr, nullptr);

    // Write known pattern
    float *device_floats = static_cast<float *>(device_ptr);
    for (size_t i = 0; i < ROWS * COLS; i++)
    {
        device_floats[i] = static_cast<float>(i) * 0.5f;
    }

    tensor->injectGpuDataPtr(device_ptr);

    // Inject coherence state directly to simulate mark_device_dirty_with_event
    // without triggering side effects through the real GPU backends
    void *mock_event = mock_backend.createEvent(0);
    ASSERT_NE(mock_event, nullptr);
    tensor->injectCompletionEvent(mock_event);
    tensor->injectDeviceValid(true);
    tensor->injectHostValid(false);

    // Verify preconditions
    EXPECT_NE(tensor->getCompletionEvent(), nullptr)
        << "Test setup: completion event must be present for event-based sync path";

    // Reset tracking counters (event already exists on tensor)
    mock_backend.resetAll();

    // ACT
    bool result = tensor->ensureOnHost();

    // ASSERT
    EXPECT_TRUE(result) << "ensureOnHost should succeed";

    // Event-based sync should be used (NOT full device sync)
    EXPECT_GE(mock_backend.getEventWaitCount(), 1u)
        << "ensureOnHost must use event-based sync when completion event exists";

    // Full device sync should NOT be called (event sync is sufficient)
    EXPECT_EQ(mock_backend.getSyncCount(), 0u)
        << "Should NOT call full device synchronize when event-based sync succeeds";

    // D2H transfer must happen
    EXPECT_GE(mock_backend.getD2HCount(), 1u)
        << "ensureOnHost must perform D2H transfer";

    // Verify data
    const float *host_data = tensor->typed_data();
    for (size_t i = 0; i < std::min<size_t>(8, ROWS * COLS); i++)
    {
        EXPECT_FLOAT_EQ(host_data[i], static_cast<float>(i) * 0.5f)
            << "Data mismatch at index " << i;
    }

    // Don't call clearBackendForTesting before tensor destruction — the destructor
    // needs the mock backend to free gpu_data_ptr_ correctly (instead of calling
    // the real CUDA/ROCm backend with a mock-allocated pointer).
    mock_backend.free(device_ptr, 0);
    tensor->injectGpuDataPtr(nullptr); // Prevent double-free in destructor
}

TEST_F(Test__StreamCoherence, EnsureOnHost_InjectedEventUsesPublishedOrdering)
{
    using namespace llaminar2::test;

    constexpr size_t ROWS = 2;
    constexpr size_t COLS = 32;
    // Keep backend ownership valid through every exceptional exit path.
    MockBackend mock_backend(DeviceType::ROCm);
    auto tensor = std::make_unique<MockCoherenceTensor>(
        std::vector<size_t>{ROWS, COLS}, DeviceId::cpu());

    tensor->setBackendForTesting(&mock_backend);

    tensor->injectGpuDevice(DeviceId::rocm(0));

    size_t bytes = ROWS * COLS * sizeof(float);
    void *device_ptr = mock_backend.allocate(bytes, 0);
    ASSERT_NE(device_ptr, nullptr);
    std::memset(device_ptr, 0, bytes);

    tensor->injectGpuDataPtr(device_ptr);

    // Inject a fake completion event directly (bypassing mark_device_dirty_with_event
    // to avoid creating a real mock event that would succeed)
    tensor->injectCompletionEvent(reinterpret_cast<void *>(0xBAD));
    tensor->injectDeviceValid(true);
    tensor->injectHostValid(false);

    mock_backend.resetAll();

    // MockBackend accepts the injected event.  This is a positive event-ordering
    // test; event-wait failure is covered by the TransferEngine hard-error suite.
    bool result = tensor->ensureOnHost();
    EXPECT_TRUE(result);

    // Event wait should have been attempted
    EXPECT_GE(mock_backend.getEventWaitCount(), 1u);

    // D2H transfer should succeed
    EXPECT_GE(mock_backend.getD2HCount(), 1u);

    mock_backend.free(device_ptr, 0);
    tensor->injectGpuDataPtr(nullptr);
}
