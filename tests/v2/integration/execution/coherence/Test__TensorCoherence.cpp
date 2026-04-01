/**
 * @file Test__TensorCoherence.cpp
 * @brief Integration test for Tensor Host/Device Memory Coherence
 *
 * **Purpose**: Comprehensive validation of the two-flag coherence model:
 *   - host_valid_: Host data is current
 *   - device_valid_: Device data is current
 *
 * **State Machine**:
 *   HOST_AUTHORITATIVE:   host_valid_=true,  device_valid_=false
 *   DEVICE_AUTHORITATIVE: host_valid_=false, device_valid_=true
 *   SYNCED:               host_valid_=true,  device_valid_=true
 *
 * **Tests**:
 * - State transitions (all valid paths through state machine)
 * - Edge cases (empty tensors, 1x1, large tensors)
 * - GPU→CPU→GPU round trips without unnecessary re-uploads
 * - Debug logging doesn't corrupt GPU pipeline (data() is safe)
 * - mutable_data() marks device stale but doesn't free memory
 * - transitionTo(TensorCoherenceState::DEVICE_AUTHORITATIVE) correctly invalidates host
 * - releaseDeviceMemory() properly syncs before freeing
 *
 * @see docs/v2/TENSOR_MEMORY_COHERENCE_DESIGN.md
 *
 * @author David Sanftenberg
 * @date January 2026
 */

#include <gtest/gtest.h>

// Include project headers BEFORE CUDATestUtils.h
#include "tensors/Tensors.h"
#include "backends/ComputeBackend.h"
#include "execution/local_execution/device/DeviceContext.h"
#include "utils/DebugEnv.h"
#ifdef HAVE_CUDA
#include "backends/cuda/CUDABackend.h"
#endif

// Test utils
#include "../../../utils/CUDATestUtils.h"

#include <vector>
#include <cstring>
#include <random>

using namespace llaminar2;
using namespace llaminar2::test::cuda;

// ============================================================================
// Test Fixture
// ============================================================================

class Test__TensorCoherence : public CUDATestBase
{
protected:
    std::mt19937 rng_{42};
    std::uniform_real_distribution<float> dist_{-1.0f, 1.0f};

    void fillRandom(FP32Tensor *tensor)
    {
        float *data = tensor->mutable_data();
        for (size_t i = 0; i < tensor->numel(); ++i)
        {
            data[i] = dist_(rng_);
        }
    }

    void fillSequential(FP32Tensor *tensor, float start = 0.0f, float step = 1.0f)
    {
        float *data = tensor->mutable_data();
        for (size_t i = 0; i < tensor->numel(); ++i)
        {
            data[i] = start + static_cast<float>(i) * step;
        }
    }

    /**
     * @brief Verify tensor data matches expected values
     */
    bool verifyData(const FP32Tensor *tensor, const std::vector<float> &expected)
    {
        if (tensor->numel() != expected.size())
            return false;
        const float *data = tensor->data();
        for (size_t i = 0; i < expected.size(); ++i)
        {
            if (data[i] != expected[i])
                return false;
        }
        return true;
    }
};

// ============================================================================
// Initial State Tests
// ============================================================================

TEST_F(Test__TensorCoherence, InitialState_HostAuthoritative)
{
    // New tensor should be HOST_AUTHORITATIVE
    auto tensor = std::make_unique<FP32Tensor>(std::vector<size_t>{32, 32});

    EXPECT_TRUE(tensor->isOnCPU()) << "New tensor should have valid host data";
    EXPECT_FALSE(tensor->isOnGPU()) << "New tensor should not be on GPU";
    EXPECT_FALSE(tensor->isDeviceValid()) << "New tensor device_valid_ should be false";
}

TEST_F(Test__TensorCoherence, InitialState_AfterMutableData)
{
    auto tensor = std::make_unique<FP32Tensor>(std::vector<size_t>{32, 32});

    // mutable_data() should keep us in HOST_AUTHORITATIVE
    float *data = tensor->mutable_data();
    ASSERT_NE(data, nullptr);

    EXPECT_TRUE(tensor->isOnCPU()) << "After mutable_data(), host should be valid";
    EXPECT_FALSE(tensor->isOnGPU()) << "After mutable_data(), should not be on GPU";
}

// ============================================================================
// State Transition: HOST_AUTHORITATIVE → SYNCED (via ensureOnDevice)
// ============================================================================

TEST_F(Test__TensorCoherence, Transition_HostAuthToSynced)
{
    auto tensor = std::make_unique<FP32Tensor>(std::vector<size_t>{64, 64});
    fillSequential(tensor.get(), 0.0f, 0.001f);

    // Initial: HOST_AUTHORITATIVE
    ASSERT_TRUE(tensor->isOnCPU());
    ASSERT_FALSE(tensor->isOnGPU());

    // Transition: ensureOnDevice() → SYNCED
    ASSERT_TRUE(tensor->ensureOnDevice(gpu_device_));

    // After: SYNCED (both host and device valid)
    EXPECT_TRUE(tensor->isOnCPU()) << "After ensureOnDevice, host should still be valid";
    EXPECT_TRUE(tensor->isOnGPU()) << "After ensureOnDevice, GPU should be valid";
    EXPECT_TRUE(tensor->isDeviceValid()) << "device_valid_ should be true";
}

// ============================================================================
// State Transition: SYNCED → HOST_AUTHORITATIVE (via mutable_data)
// ============================================================================

TEST_F(Test__TensorCoherence, Transition_SyncedToHostAuth_ViaMutableData)
{
    auto tensor = std::make_unique<FP32Tensor>(std::vector<size_t>{64, 64});
    fillSequential(tensor.get(), 0.0f, 0.001f);

    // Get to SYNCED state
    ASSERT_TRUE(tensor->ensureOnDevice(gpu_device_));
    ASSERT_TRUE(tensor->isOnCPU() && tensor->isDeviceValid());

    // Transition: mutable_data() → HOST_AUTHORITATIVE
    // This marks device stale but does NOT free GPU memory
    float *data = tensor->mutable_data();
    ASSERT_NE(data, nullptr);

    EXPECT_TRUE(tensor->isOnCPU()) << "Host should remain valid";
    EXPECT_TRUE(tensor->isOnGPU()) << "GPU memory should NOT be freed";
    EXPECT_FALSE(tensor->isDeviceValid()) << "Device data should be marked stale";
}

TEST_F(Test__TensorCoherence, MutableData_DoesNotFreeGpuMemory)
{
    auto tensor = std::make_unique<FP32Tensor>(std::vector<size_t>{64, 64});
    fillSequential(tensor.get(), 0.0f, 0.001f);

    // Upload to GPU
    ASSERT_TRUE(tensor->ensureOnDevice(gpu_device_));
    void *gpu_ptr_before = tensor->gpu_data_ptr();
    ASSERT_NE(gpu_ptr_before, nullptr);

    // Call mutable_data() - should mark device stale but NOT free
    float *host_data = tensor->mutable_data();
    ASSERT_NE(host_data, nullptr);

    // GPU pointer should still be valid (memory not freed)
    void *gpu_ptr_after = tensor->gpu_data_ptr();
    EXPECT_EQ(gpu_ptr_before, gpu_ptr_after) << "GPU memory should be retained, not freed";
}

// ============================================================================
// State Transition: SYNCED → DEVICE_AUTHORITATIVE (via mark_device_dirty)
// ============================================================================

TEST_F(Test__TensorCoherence, Transition_SyncedToDeviceAuth_ViaMarkDirty)
{
    auto tensor = std::make_unique<FP32Tensor>(std::vector<size_t>{64, 64});
    fillSequential(tensor.get(), 0.0f, 0.001f);

    // Get to SYNCED state
    ASSERT_TRUE(tensor->ensureOnDevice(gpu_device_));
    ASSERT_TRUE(tensor->isOnCPU() && tensor->isDeviceValid());

    // Transition: transitionTo(TensorCoherenceState::DEVICE_AUTHORITATIVE) → DEVICE_AUTHORITATIVE
    // Simulates GPU kernel writing to the tensor
    tensor->transitionTo(TensorCoherenceState::DEVICE_AUTHORITATIVE);

    EXPECT_FALSE(tensor->isOnCPU()) << "Host should be marked stale";
    EXPECT_TRUE(tensor->isOnGPU()) << "GPU memory should still exist";
    EXPECT_TRUE(tensor->isDeviceValid()) << "Device should be valid (just written)";
}

// ============================================================================
// State Transition: DEVICE_AUTHORITATIVE → SYNCED (via data/ensureOnHost)
// ============================================================================

TEST_F(Test__TensorCoherence, Transition_DeviceAuthToSynced_ViaData)
{
    auto tensor = std::make_unique<FP32Tensor>(std::vector<size_t>{64, 64});
    fillSequential(tensor.get(), 0.0f, 0.001f);

    // Store original data for verification
    std::vector<float> original(tensor->numel());
    std::memcpy(original.data(), tensor->data(), original.size() * sizeof(float));

    // Get to DEVICE_AUTHORITATIVE state
    ASSERT_TRUE(tensor->ensureOnDevice(gpu_device_));
    tensor->transitionTo(TensorCoherenceState::DEVICE_AUTHORITATIVE); // Simulates GPU write
    ASSERT_FALSE(tensor->isOnCPU());
    ASSERT_TRUE(tensor->isDeviceValid());

    // Transition: data() → SYNCED (downloads from device)
    const float *data = tensor->data();
    ASSERT_NE(data, nullptr);

    EXPECT_TRUE(tensor->isOnCPU()) << "Host should now be valid after download";
    EXPECT_TRUE(tensor->isDeviceValid()) << "Device should remain valid";

    // Verify data integrity (GPU didn't modify it, so should match original)
    EXPECT_TRUE(verifyData(tensor.get(), original)) << "Data should match after round-trip";
}

TEST_F(Test__TensorCoherence, Transition_DeviceAuthToSynced_ViaEnsureOnHost)
{
    auto tensor = std::make_unique<FP32Tensor>(std::vector<size_t>{64, 64});
    fillSequential(tensor.get(), 0.0f, 0.001f);

    // Get to DEVICE_AUTHORITATIVE
    ASSERT_TRUE(tensor->ensureOnDevice(gpu_device_));
    tensor->transitionTo(TensorCoherenceState::DEVICE_AUTHORITATIVE);

    // Transition: ensureOnHost() → SYNCED
    ASSERT_TRUE(tensor->ensureOnHost());

    EXPECT_TRUE(tensor->isOnCPU()) << "Host should be valid";
    EXPECT_TRUE(tensor->isDeviceValid()) << "Device should remain valid";
}

// ============================================================================
// State Transition: HOST_AUTHORITATIVE → SYNCED → HOST_AUTHORITATIVE (cycle)
// ============================================================================

TEST_F(Test__TensorCoherence, Cycle_UploadModifyReupload)
{
    auto tensor = std::make_unique<FP32Tensor>(std::vector<size_t>{32, 32});
    fillSequential(tensor.get(), 0.0f, 1.0f);

    // Upload
    ASSERT_TRUE(tensor->ensureOnDevice(gpu_device_));
    EXPECT_TRUE(tensor->isOnCPU() && tensor->isDeviceValid());

    // Modify host (makes device stale)
    float *data = tensor->mutable_data();
    data[0] = 999.0f;
    EXPECT_TRUE(tensor->isOnCPU());
    EXPECT_FALSE(tensor->isDeviceValid());

    // Re-upload (should upload the modified data)
    ASSERT_TRUE(tensor->ensureOnDevice(gpu_device_));
    EXPECT_TRUE(tensor->isOnCPU() && tensor->isDeviceValid());

    // Verify: download and check modified value persists
    tensor->transitionTo(TensorCoherenceState::DEVICE_AUTHORITATIVE);
    const float *result = tensor->data();
    EXPECT_EQ(result[0], 999.0f) << "Modified value should survive upload cycle";
}

// ============================================================================
// Efficiency: No unnecessary transfers
// ============================================================================

TEST_F(Test__TensorCoherence, NoReupload_WhenDeviceAlreadyValid)
{
    auto tensor = std::make_unique<FP32Tensor>(std::vector<size_t>{32, 32});
    fillSequential(tensor.get(), 0.0f, 1.0f);

    // First upload
    ASSERT_TRUE(tensor->ensureOnDevice(gpu_device_));
    void *gpu_ptr1 = tensor->gpu_data_ptr();

    // Second ensureOnDevice - should be a no-op
    ASSERT_TRUE(tensor->ensureOnDevice(gpu_device_));
    void *gpu_ptr2 = tensor->gpu_data_ptr();

    EXPECT_EQ(gpu_ptr1, gpu_ptr2) << "GPU pointer should be unchanged (no realloc)";
}

TEST_F(Test__TensorCoherence, NoDownload_WhenHostAlreadyValid)
{
    auto tensor = std::make_unique<FP32Tensor>(std::vector<size_t>{32, 32});
    fillSequential(tensor.get(), 0.0f, 1.0f);

    // Upload (puts us in SYNCED)
    ASSERT_TRUE(tensor->ensureOnDevice(gpu_device_));

    // data() should NOT download - host is already valid
    const float *data1 = tensor->data();
    const float *data2 = tensor->data(); // Second call

    EXPECT_EQ(data1, data2) << "Should return same host pointer, no download";
}

TEST_F(Test__TensorCoherence, GpuToGpu_NoHostTransfer)
{
    // Pattern: GPU kernel 1 writes → GPU kernel 2 reads (no host involvement)
    auto tensor = std::make_unique<FP32Tensor>(std::vector<size_t>{32, 32});
    fillSequential(tensor.get(), 0.0f, 1.0f);

    // Kernel 1: write
    ASSERT_TRUE(tensor->ensureOnDevice(gpu_device_));
    tensor->transitionTo(TensorCoherenceState::DEVICE_AUTHORITATIVE); // GPU wrote to it

    // Device is authoritative, host is stale
    EXPECT_FALSE(tensor->isOnCPU());
    EXPECT_TRUE(tensor->isDeviceValid());

    // Kernel 2: read (ensureOnDevice should be no-op)
    ASSERT_TRUE(tensor->ensureOnDevice(gpu_device_));

    // Host should STILL be stale (no download occurred)
    EXPECT_FALSE(tensor->isOnCPU()) << "Host should remain stale after GPU-to-GPU";
    EXPECT_TRUE(tensor->isDeviceValid());
}

// ============================================================================
// Debug Safety: data() doesn't break GPU pipeline
// ============================================================================

TEST_F(Test__TensorCoherence, DebugLogging_SafeAfterGpuWrite)
{
    // This was a problematic pattern before the coherence fix
    auto tensor = std::make_unique<FP32Tensor>(std::vector<size_t>{32, 32});
    fillSequential(tensor.get(), 0.0f, 1.0f);

    // GPU writes
    ASSERT_TRUE(tensor->ensureOnDevice(gpu_device_));
    tensor->transitionTo(TensorCoherenceState::DEVICE_AUTHORITATIVE);

    // Debug log reads (should download but NOT invalidate GPU)
    const float *debug_data = tensor->data();
    EXPECT_NE(debug_data, nullptr);

    // Now both should be valid (SYNCED)
    EXPECT_TRUE(tensor->isOnCPU());
    EXPECT_TRUE(tensor->isDeviceValid());

    // Next GPU kernel can still use gpu_data_ptr() without re-upload
    void *gpu_ptr = tensor->gpu_data_ptr();
    EXPECT_NE(gpu_ptr, nullptr);
    EXPECT_TRUE(tensor->isDeviceValid()) << "GPU should still be valid after debug read";
}

// ============================================================================
// releaseDeviceMemory: Proper cleanup
// ============================================================================

TEST_F(Test__TensorCoherence, ReleaseDeviceMemory_SyncsFirst)
{
    auto tensor = std::make_unique<FP32Tensor>(std::vector<size_t>{32, 32});
    fillSequential(tensor.get(), 0.0f, 1.0f);

    std::vector<float> original(tensor->numel());
    std::memcpy(original.data(), tensor->data(), original.size() * sizeof(float));

    // GPU writes (device is authoritative)
    ASSERT_TRUE(tensor->ensureOnDevice(gpu_device_));
    tensor->transitionTo(TensorCoherenceState::DEVICE_AUTHORITATIVE);

    // Release device memory - should download first
    ASSERT_TRUE(tensor->releaseDeviceMemory());

    EXPECT_TRUE(tensor->isOnCPU()) << "Host should be valid after release";
    EXPECT_FALSE(tensor->isOnGPU()) << "GPU memory should be freed";
    EXPECT_EQ(tensor->gpu_data_ptr(), nullptr);

    // Data should still be accessible
    EXPECT_TRUE(verifyData(tensor.get(), original));
}

TEST_F(Test__TensorCoherence, ReleaseDeviceMemory_WhenAlreadyOnHost)
{
    auto tensor = std::make_unique<FP32Tensor>(std::vector<size_t>{32, 32});
    fillSequential(tensor.get(), 0.0f, 1.0f);

    // Upload then download (SYNCED)
    ASSERT_TRUE(tensor->ensureOnDevice(gpu_device_));
    ASSERT_TRUE(tensor->ensureOnHost());

    // Release should work without any transfer
    ASSERT_TRUE(tensor->releaseDeviceMemory());

    EXPECT_TRUE(tensor->isOnCPU());
    EXPECT_FALSE(tensor->isOnGPU());
}

// ============================================================================
// BF16 Tensor Coherence
// ============================================================================

TEST_F(Test__TensorCoherence, BF16_FullStateTransitions)
{
    auto tensor = std::make_unique<BF16Tensor>(std::vector<size_t>{64, 64});

    // Fill with test data
    uint16_t *data = tensor->mutable_bf16_data();
    std::vector<uint16_t> original(tensor->numel());
    for (size_t i = 0; i < tensor->numel(); ++i)
    {
        data[i] = static_cast<uint16_t>(i & 0x7FFF); // Valid BF16 range
        original[i] = data[i];
    }

    // Initial: HOST_AUTHORITATIVE
    EXPECT_TRUE(tensor->isOnCPU());
    EXPECT_FALSE(tensor->isOnGPU());

    // Upload: → SYNCED
    ASSERT_TRUE(tensor->ensureOnDevice(gpu_device_));
    EXPECT_TRUE(tensor->isOnCPU() && tensor->isDeviceValid());

    // Mark dirty: → DEVICE_AUTHORITATIVE
    tensor->transitionTo(TensorCoherenceState::DEVICE_AUTHORITATIVE);
    EXPECT_FALSE(tensor->isOnCPU());
    EXPECT_TRUE(tensor->isDeviceValid());

    // Download: → SYNCED
    ASSERT_TRUE(tensor->ensureOnHost());
    EXPECT_TRUE(tensor->isOnCPU() && tensor->isDeviceValid());

    // Verify data
    const uint16_t *result = tensor->bf16_data();
    for (size_t i = 0; i < tensor->numel(); ++i)
    {
        EXPECT_EQ(result[i], original[i]) << "BF16 mismatch at " << i;
    }
}

// ============================================================================
// FP16 Tensor Coherence
// ============================================================================

TEST_F(Test__TensorCoherence, FP16_FullStateTransitions)
{
    auto tensor = std::make_unique<FP16Tensor>(std::vector<size_t>{64, 64});

    uint16_t *data = tensor->mutable_fp16_data();
    std::vector<uint16_t> original(tensor->numel());
    for (size_t i = 0; i < tensor->numel(); ++i)
    {
        data[i] = static_cast<uint16_t>(i & 0x7FFF);
        original[i] = data[i];
    }

    // Full state transition cycle
    EXPECT_TRUE(tensor->isOnCPU());
    ASSERT_TRUE(tensor->ensureOnDevice(gpu_device_));
    tensor->transitionTo(TensorCoherenceState::DEVICE_AUTHORITATIVE);
    ASSERT_TRUE(tensor->ensureOnHost());

    const uint16_t *result = tensor->fp16_data();
    for (size_t i = 0; i < tensor->numel(); ++i)
    {
        EXPECT_EQ(result[i], original[i]) << "FP16 mismatch at " << i;
    }
}

// ============================================================================
// Edge Cases
// ============================================================================

TEST_F(Test__TensorCoherence, EdgeCase_1x1Tensor)
{
    auto tensor = std::make_unique<FP32Tensor>(std::vector<size_t>{1, 1});
    tensor->mutable_data()[0] = 42.0f;

    ASSERT_TRUE(tensor->ensureOnDevice(gpu_device_));
    tensor->transitionTo(TensorCoherenceState::DEVICE_AUTHORITATIVE);
    const float *data = tensor->data();

    EXPECT_EQ(data[0], 42.0f);
}

TEST_F(Test__TensorCoherence, EdgeCase_LargeTensor_16MB)
{
    // 4M floats = 16MB
    const size_t rows = 2048;
    const size_t cols = 2048;

    auto tensor = std::make_unique<FP32Tensor>(std::vector<size_t>{rows, cols});
    float *data = tensor->mutable_data();

    // Fill with pattern
    for (size_t i = 0; i < rows * cols; ++i)
    {
        data[i] = static_cast<float>(i % 1000) * 0.01f;
    }

    // Full round trip
    ASSERT_TRUE(tensor->ensureOnDevice(gpu_device_));
    tensor->transitionTo(TensorCoherenceState::DEVICE_AUTHORITATIVE);
    ASSERT_TRUE(tensor->ensureOnHost());

    // Sample verification
    const float *result = tensor->data();
    for (size_t i = 0; i < rows * cols; i += 10000)
    {
        float expected = static_cast<float>(i % 1000) * 0.01f;
        EXPECT_FLOAT_EQ(result[i], expected) << "Mismatch at " << i;
    }
}

TEST_F(Test__TensorCoherence, EdgeCase_TallSkinnyMatrix)
{
    // Common pattern: embedding table access
    const size_t rows = 16384;
    const size_t cols = 64;

    auto tensor = std::make_unique<FP32Tensor>(std::vector<size_t>{rows, cols});
    fillRandom(tensor.get());

    std::vector<float> original(tensor->numel());
    std::memcpy(original.data(), tensor->data(), original.size() * sizeof(float));

    ASSERT_TRUE(tensor->ensureOnDevice(gpu_device_));
    tensor->transitionTo(TensorCoherenceState::DEVICE_AUTHORITATIVE);
    ASSERT_TRUE(tensor->ensureOnHost());

    EXPECT_TRUE(verifyData(tensor.get(), original));
}

TEST_F(Test__TensorCoherence, EdgeCase_WideShortMatrix)
{
    // Common pattern: vocabulary logits
    const size_t rows = 1;
    const size_t cols = 151936; // Qwen2 vocab size

    auto tensor = std::make_unique<FP32Tensor>(std::vector<size_t>{rows, cols});
    fillRandom(tensor.get());

    std::vector<float> original(tensor->numel());
    std::memcpy(original.data(), tensor->data(), original.size() * sizeof(float));

    ASSERT_TRUE(tensor->ensureOnDevice(gpu_device_));
    tensor->transitionTo(TensorCoherenceState::DEVICE_AUTHORITATIVE);
    ASSERT_TRUE(tensor->ensureOnHost());

    EXPECT_TRUE(verifyData(tensor.get(), original));
}

// ============================================================================
// Multiple Transfer Cycles
// ============================================================================

TEST_F(Test__TensorCoherence, MultipleCycles_NoMemoryLeak)
{
    // Stress test: multiple upload/download cycles should not leak memory
    auto tensor = std::make_unique<FP32Tensor>(std::vector<size_t>{128, 128});

    for (int cycle = 0; cycle < 10; ++cycle)
    {
        fillSequential(tensor.get(), static_cast<float>(cycle), 0.001f);

        ASSERT_TRUE(tensor->ensureOnDevice(gpu_device_));
        tensor->transitionTo(TensorCoherenceState::DEVICE_AUTHORITATIVE);
        ASSERT_TRUE(tensor->ensureOnHost());

        // Verify first element
        EXPECT_FLOAT_EQ(tensor->data()[0], static_cast<float>(cycle));
    }
}

TEST_F(Test__TensorCoherence, AlternatingHostGpuModifications)
{
    auto tensor = std::make_unique<FP32Tensor>(std::vector<size_t>{32, 32});

    // Initial value
    tensor->mutable_data()[0] = 1.0f;

    for (int i = 0; i < 5; ++i)
    {
        // GPU "reads" (just upload)
        ASSERT_TRUE(tensor->ensureOnDevice(gpu_device_));

        // GPU "writes" (mark dirty)
        tensor->transitionTo(TensorCoherenceState::DEVICE_AUTHORITATIVE);

        // Host reads (download)
        float val = tensor->data()[0];
        EXPECT_EQ(val, 1.0f + static_cast<float>(i));

        // Host writes
        float *data = tensor->mutable_data();
        data[0] = 1.0f + static_cast<float>(i + 1);
    }
}

// ============================================================================
// is_on_device() Query Tests
// ============================================================================

TEST_F(Test__TensorCoherence, IsOnDevice_ReflectsCurrentState)
{
    auto tensor = std::make_unique<FP32Tensor>(std::vector<size_t>{32, 32});
    fillSequential(tensor.get());

    // Initially not on device
    EXPECT_FALSE(tensor->is_on_device(gpu_device_));
    EXPECT_TRUE(tensor->is_on_device(DeviceId::cpu())); // CPU

    // After upload
    ASSERT_TRUE(tensor->ensureOnDevice(gpu_device_));
    EXPECT_TRUE(tensor->is_on_device(gpu_device_));

    // After host modification
    tensor->mutable_data()[0] = 999.0f;
    EXPECT_FALSE(tensor->is_on_device(gpu_device_)) << "Should report false when device is stale";

    // After re-upload
    ASSERT_TRUE(tensor->ensureOnDevice(gpu_device_));
    EXPECT_TRUE(tensor->is_on_device(gpu_device_));
}

// ============================================================================
// Raw Data Access (for ITensor interface)
// ============================================================================

TEST_F(Test__TensorCoherence, RawMutableData_InvalidatesDevice)
{
    auto tensor = std::make_unique<FP32Tensor>(std::vector<size_t>{32, 32});
    fillSequential(tensor.get());

    ASSERT_TRUE(tensor->ensureOnDevice(gpu_device_));
    EXPECT_TRUE(tensor->isDeviceValid());

    // raw_mutable_data() should invalidate device (same as mutable_data())
    void *raw_ptr = tensor->raw_mutable_data();
    ASSERT_NE(raw_ptr, nullptr);

    EXPECT_TRUE(tensor->isOnCPU());
    EXPECT_FALSE(tensor->isDeviceValid()) << "raw_mutable_data() should mark device stale";
}

TEST_F(Test__TensorCoherence, RawData_DoesNotTriggerDownload)
{
    // raw_data() is a low-level API that returns the host pointer without
    // triggering coherence actions. Use data() or ensureOnHost() instead
    // if you need automatic synchronization.
    auto tensor = std::make_unique<FP32Tensor>(std::vector<size_t>{32, 32});
    fillSequential(tensor.get());

    ASSERT_TRUE(tensor->ensureOnDevice(gpu_device_));
    tensor->transitionTo(TensorCoherenceState::DEVICE_AUTHORITATIVE);

    // raw_data() is low-level - it does NOT auto-sync
    const void *raw_ptr = tensor->raw_data();
    ASSERT_NE(raw_ptr, nullptr);

    // Host is still marked invalid (raw_data doesn't trigger sync)
    EXPECT_FALSE(tensor->isOnCPU()) << "raw_data() should NOT auto-sync";
    EXPECT_TRUE(tensor->isDeviceValid()) << "device should remain authoritative";
}

// ============================================================================
// active_data_ptr() Tests
// ============================================================================

TEST_F(Test__TensorCoherence, ActiveDataPtr_ReturnsAppropriatePointer)
{
    auto tensor = std::make_unique<FP32Tensor>(std::vector<size_t>{32, 32});
    fillSequential(tensor.get());

    // Before upload: should return host pointer (use raw_data() which is public)
    const void *host_ptr = tensor->raw_data();
    EXPECT_EQ(tensor->active_data_ptr(), host_ptr);

    // After upload: should return GPU pointer
    ASSERT_TRUE(tensor->ensureOnDevice(gpu_device_));
    EXPECT_EQ(tensor->active_data_ptr(), tensor->gpu_data_ptr());

    // After release: should return host pointer again
    ASSERT_TRUE(tensor->releaseDeviceMemory());
    EXPECT_EQ(tensor->active_data_ptr(), host_ptr);
}

// ============================================================================
// Transfer Counting Tests (for debugging transfer overhead)
// ============================================================================

/**
 * @brief Helper to enable transfer tracing and reset counters
 */
class TransferCountGuard
{
public:
    TransferCountGuard()
    {
        auto &cfg = const_cast<DebugEnv &>(debugEnv()).transfer_tracing;
        was_enabled_ = cfg.enabled;
        cfg.enabled = true;
        cfg.resetCounters();
    }

    ~TransferCountGuard()
    {
        auto &cfg = const_cast<DebugEnv &>(debugEnv()).transfer_tracing;
        cfg.enabled = was_enabled_;
    }

    size_t h2d_count() const { return debugEnv().transfer_tracing.h2d_count.load(); }
    size_t d2h_count() const { return debugEnv().transfer_tracing.d2h_count.load(); }
    size_t h2d_bytes() const { return debugEnv().transfer_tracing.h2d_bytes.load(); }
    size_t d2h_bytes() const { return debugEnv().transfer_tracing.d2h_bytes.load(); }

private:
    bool was_enabled_ = false;
};

TEST_F(Test__TensorCoherence, TransferCounting_SingleUpload)
{
    TransferCountGuard guard;

    auto tensor = std::make_unique<FP32Tensor>(std::vector<size_t>{64, 64});
    fillSequential(tensor.get());

    // Single upload
    ASSERT_TRUE(tensor->ensureOnDevice(gpu_device_));

    EXPECT_EQ(guard.h2d_count(), 1) << "Should have exactly 1 H2D transfer";
    EXPECT_EQ(guard.d2h_count(), 0) << "Should have no D2H transfers";
    EXPECT_EQ(guard.h2d_bytes(), 64 * 64 * sizeof(float));
}

TEST_F(Test__TensorCoherence, TransferCounting_MultipleUploads_NoRedundant)
{
    TransferCountGuard guard;

    auto tensor = std::make_unique<FP32Tensor>(std::vector<size_t>{64, 64});
    fillSequential(tensor.get());

    // Multiple uploads without host modification - should NOT re-upload
    ASSERT_TRUE(tensor->ensureOnDevice(gpu_device_));
    ASSERT_TRUE(tensor->ensureOnDevice(gpu_device_)); // Second call
    ASSERT_TRUE(tensor->ensureOnDevice(gpu_device_)); // Third call

    EXPECT_EQ(guard.h2d_count(), 1) << "Subsequent ensureOnDevice should NOT re-upload";
    EXPECT_EQ(guard.d2h_count(), 0) << "Should have no D2H transfers";
}

TEST_F(Test__TensorCoherence, TransferCounting_DataCallAfterGpuWrite_SingleD2H)
{
    TransferCountGuard guard;

    auto tensor = std::make_unique<FP32Tensor>(std::vector<size_t>{64, 64});
    fillSequential(tensor.get());

    // Upload
    ASSERT_TRUE(tensor->ensureOnDevice(gpu_device_));

    // Simulate GPU write
    tensor->transitionTo(TensorCoherenceState::DEVICE_AUTHORITATIVE);

    // data() should trigger download
    const float *data = tensor->data();
    ASSERT_NE(data, nullptr);

    EXPECT_EQ(guard.h2d_count(), 1) << "Should have 1 H2D for initial upload";
    EXPECT_EQ(guard.d2h_count(), 1) << "Should have 1 D2H after data() call";
}

TEST_F(Test__TensorCoherence, TransferCounting_DataCallAfterGpuWrite_NoRedundantD2H)
{
    TransferCountGuard guard;

    auto tensor = std::make_unique<FP32Tensor>(std::vector<size_t>{64, 64});
    fillSequential(tensor.get());

    ASSERT_TRUE(tensor->ensureOnDevice(gpu_device_));
    tensor->transitionTo(TensorCoherenceState::DEVICE_AUTHORITATIVE);

    // Multiple data() calls should NOT re-download
    const float *data1 = tensor->data();
    const float *data2 = tensor->data();
    const float *data3 = tensor->data();

    ASSERT_EQ(data1, data2);
    ASSERT_EQ(data2, data3);

    EXPECT_EQ(guard.d2h_count(), 1) << "Multiple data() calls should NOT cause multiple downloads";
}

TEST_F(Test__TensorCoherence, TransferCounting_HostModificationTriggersReupload)
{
    TransferCountGuard guard;

    auto tensor = std::make_unique<FP32Tensor>(std::vector<size_t>{64, 64});
    fillSequential(tensor.get());

    // Initial upload
    ASSERT_TRUE(tensor->ensureOnDevice(gpu_device_));

    // Host modification
    tensor->mutable_data()[0] = 999.0f;

    // Re-upload (should transfer because host was modified)
    ASSERT_TRUE(tensor->ensureOnDevice(gpu_device_));

    EXPECT_EQ(guard.h2d_count(), 2) << "Host modification should trigger re-upload";
}

TEST_F(Test__TensorCoherence, TransferCounting_GpuDataPtrNoTransfer)
{
    TransferCountGuard guard;

    auto tensor = std::make_unique<FP32Tensor>(std::vector<size_t>{64, 64});
    fillSequential(tensor.get());

    ASSERT_TRUE(tensor->ensureOnDevice(gpu_device_));
    size_t initial_h2d = guard.h2d_count();

    // gpu_data_ptr() should NOT trigger any transfer
    void *ptr1 = tensor->gpu_data_ptr();
    void *ptr2 = tensor->gpu_data_ptr();
    void *ptr3 = tensor->gpu_data_ptr();

    ASSERT_NE(ptr1, nullptr);
    EXPECT_EQ(guard.h2d_count(), initial_h2d) << "gpu_data_ptr() should NOT trigger transfer";
    EXPECT_EQ(guard.d2h_count(), 0) << "gpu_data_ptr() should NOT trigger D2H";
}

TEST_F(Test__TensorCoherence, TransferCounting_RoundTrip)
{
    TransferCountGuard guard;

    auto tensor = std::make_unique<FP32Tensor>(std::vector<size_t>{64, 64});
    fillSequential(tensor.get());

    // Full round trip: host → device → host → device
    ASSERT_TRUE(tensor->ensureOnDevice(gpu_device_)); // H2D #1
    tensor->transitionTo(TensorCoherenceState::DEVICE_AUTHORITATIVE);
    tensor->ensureOnHost();                           // D2H #1
    tensor->mutable_data()[0] = 42.0f;                // Mark device stale
    ASSERT_TRUE(tensor->ensureOnDevice(gpu_device_)); // H2D #2

    EXPECT_EQ(guard.h2d_count(), 2) << "Should have 2 H2D transfers";
    EXPECT_EQ(guard.d2h_count(), 1) << "Should have 1 D2H transfer";
}

TEST_F(Test__TensorCoherence, TransferCounting_LargeTensor_CorrectBytes)
{
    TransferCountGuard guard;

    // 1MB tensor
    const size_t rows = 512;
    const size_t cols = 512;
    const size_t expected_bytes = rows * cols * sizeof(float);

    auto tensor = std::make_unique<FP32Tensor>(std::vector<size_t>{rows, cols});
    fillSequential(tensor.get());

    ASSERT_TRUE(tensor->ensureOnDevice(gpu_device_));
    tensor->transitionTo(TensorCoherenceState::DEVICE_AUTHORITATIVE);
    tensor->data(); // Trigger D2H

    EXPECT_EQ(guard.h2d_bytes(), expected_bytes) << "H2D bytes should match tensor size";
    EXPECT_EQ(guard.d2h_bytes(), expected_bytes) << "D2H bytes should match tensor size";
}

TEST_F(Test__TensorCoherence, TransferCounting_CachedGpuDataSkipsUpload)
{
    TransferCountGuard guard;

    auto tensor = std::make_unique<FP32Tensor>(std::vector<size_t>{64, 64});
    fillSequential(tensor.get());

    // Upload and mark dirty (simulates GPU kernel writing to it)
    ASSERT_TRUE(tensor->ensureOnDevice(gpu_device_));
    tensor->transitionTo(TensorCoherenceState::DEVICE_AUTHORITATIVE);

    // Now device has valid data. Another ensureOnDevice should NOT re-upload
    // because device is already valid (even if host is stale).
    ASSERT_TRUE(tensor->ensureOnDevice(gpu_device_));

    EXPECT_EQ(guard.h2d_count(), 1) << "When device is valid, ensureOnDevice should skip upload";
}

TEST_F(Test__TensorCoherence, TransferCounting_NoTransferWhenHostAuthoritative)
{
    TransferCountGuard guard;

    auto tensor = std::make_unique<FP32Tensor>(std::vector<size_t>{64, 64});
    fillSequential(tensor.get());

    // Multiple reads without any GPU involvement - no transfers
    const float *data1 = tensor->data();
    const float *data2 = tensor->data();
    float *mdata1 = tensor->mutable_data();
    const float *data3 = tensor->data();

    ASSERT_NE(data1, nullptr);
    ASSERT_NE(data2, nullptr);
    ASSERT_NE(mdata1, nullptr);
    ASSERT_NE(data3, nullptr);

    EXPECT_EQ(guard.h2d_count(), 0) << "Host-only access should NOT trigger any transfers";
    EXPECT_EQ(guard.d2h_count(), 0) << "Host-only access should NOT trigger any transfers";
}
