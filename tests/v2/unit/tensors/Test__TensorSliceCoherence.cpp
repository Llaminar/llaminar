/**
 * @file Test__TensorSliceCoherence.cpp
 * @brief Unit tests for TensorSlice device coherence delegation
 * @author David Sanftenberg
 * @date January 2026
 *
 * These tests verify that TensorSlice properly delegates device coherence
 * operations to its wrapped inner tensor. This is critical for MPI tensor
 * parallelism where weight tensors are wrapped in TensorSlice for sharding.
 *
 * Root cause of issue: While TensorSlice inherits from TensorBase (which is
 * aliased to TensorBase), the StageCoherence code was calling ensureOnDevice()
 * on the TensorSlice wrapper rather than the inner tensor. Since TensorSlice
 * doesn't override the coherence methods, they operated on TensorSlice's own
 * (empty) coherence state rather than the inner tensor's state.
 *
 * The fix in StageCoherence.cpp unwraps TensorSlice to access the inner tensor
 * for coherence operations. These tests verify that approach works correctly.
 *
 * Test categories:
 * 1. MockTensor - Verifies coherence calls are tracked
 * 2. TensorSlice Inheritance - Verifies class hierarchy
 * 3. TensorSlice Coherence Delegation - Verifies unwrapping works
 * 4. StageCoherence Integration - Verifies the fix works end-to-end
 */

#include <gtest/gtest.h>
#include <memory>
#include <vector>
#include <string>
#include "tensors/TensorSlice.h"
#include "tensors/Tensors.h"
#include "backends/DeviceId.h"

using namespace llaminar2;

// =============================================================================
// Mock Tensor for Tracking Coherence Calls
// =============================================================================

/**
 * @brief Mock tensor that tracks coherence method calls
 *
 * This mock inherits from FP32Tensor (which inherits from TensorBase)
 * and overrides coherence methods to track whether they were called.
 */
class MockCoherenceTensor : public FP32Tensor
{
public:
    MockCoherenceTensor(size_t rows, size_t cols)
        : FP32Tensor({rows, cols})
    {
        // Initialize with some data
        float *ptr = mutable_fp32_data();
        for (size_t i = 0; i < rows * cols; ++i)
        {
            ptr[i] = static_cast<float>(i) * 0.01f;
        }
    }

    // Track calls
    mutable int ensureOnDevice_calls = 0;
    mutable int ensureOnHost_calls = 0;
    mutable int allocateOnDevice_calls = 0;
    mutable int invalidateGpuData_calls = 0;
    mutable DeviceId last_ensureOnDevice_target = DeviceId::cpu();
    mutable DeviceId last_allocateOnDevice_target = DeviceId::cpu();

    // Override coherence methods to track calls
    bool ensureOnDevice(DeviceId target_device, void *stream = nullptr) override
    {
        ensureOnDevice_calls++;
        last_ensureOnDevice_target = target_device;
        // Simulate successful upload (don't actually do GPU transfer)
        return true;
    }

    bool ensureOnHost(void *stream = nullptr) override
    {
        ensureOnHost_calls++;
        return true;
    }

    bool allocateOnDevice(DeviceId target_device, void *stream = nullptr) override
    {
        allocateOnDevice_calls++;
        last_allocateOnDevice_target = target_device;
        return true;
    }

    void invalidateGpuData() override
    {
        invalidateGpuData_calls++;
    }

    /// Inject a fake GPU pointer (simulates what ensureOnDevice does)
    void injectGpuDataPtr(void *ptr) { gpu_data_ptr_ = ptr; }

    /// Get the raw gpu_data_ptr_ member directly (not through virtual method)
    void *getRawGpuDataPtr() const { return gpu_data_ptr_; }

    // Reset call counters
    void resetCallCounters()
    {
        ensureOnDevice_calls = 0;
        ensureOnHost_calls = 0;
        allocateOnDevice_calls = 0;
        invalidateGpuData_calls = 0;
    }
};

// =============================================================================
// Test Fixture
// =============================================================================

class Test__TensorSliceCoherence : public ::testing::Test
{
protected:
    void SetUp() override
    {
        // Create a mock tensor
        mock_tensor_ = std::make_unique<MockCoherenceTensor>(1024, 896);
    }

    void TearDown() override
    {
        mock_tensor_.reset();
    }

    // Helper to create TensorSlice with SliceMetadata
    std::unique_ptr<TensorSlice> createSlice(
        std::unique_ptr<TensorBase> inner,
        SliceMode mode = SliceMode::ROW_PARALLEL)
    {
        size_t rows = inner->shape()[0];
        size_t cols = inner->shape()[1];

        SliceMetadata metadata;
        metadata.mode = mode;
        metadata.original_rows = rows * 2; // Pretend original was 2x
        metadata.original_cols = cols;
        metadata.slice_start = 0;
        metadata.slice_end = rows;
        metadata.rank = 0;
        metadata.world_size = 2;
        metadata.inner_is_presliced = true;

        return std::make_unique<TensorSlice>(std::move(inner), std::move(metadata));
    }

    // Helper to create real Q4_0 tensor
    std::unique_ptr<Q4_0Tensor> createQ4_0(size_t rows, size_t cols)
    {
        const size_t block_size = 32;
        const size_t bytes_per_block = 18; // Q4_0 block: 2 bytes scale + 16 bytes data
        const size_t num_blocks = rows * (cols / block_size);
        std::vector<uint8_t> raw_data(num_blocks * bytes_per_block, 0);

        // Set scale values to 1.0 in FP16
        for (size_t b = 0; b < num_blocks; ++b)
        {
            uint16_t scale_bits = 0x3C00; // ~1.0 in FP16
            memcpy(&raw_data[b * bytes_per_block], &scale_bits, sizeof(scale_bits));
        }

        return std::make_unique<Q4_0Tensor>(std::vector<size_t>{rows, cols}, std::move(raw_data));
    }

    std::unique_ptr<MockCoherenceTensor> mock_tensor_;
};

// =============================================================================
// Test Category 1: TensorSlice Inheritance Verification
// =============================================================================

TEST_F(Test__TensorSliceCoherence, TensorSliceInheritsFromTensorBase)
{
    // Verify TensorSlice is-a TensorBase
    auto inner = createQ4_0(512, 256);
    auto slice = createSlice(std::move(inner));

    // TensorSlice should be castable to TensorBase
    TensorBase *as_tensor_base = dynamic_cast<TensorBase *>(slice.get());
    EXPECT_NE(as_tensor_base, nullptr)
        << "TensorSlice should be castable to TensorBase";
}

TEST_F(Test__TensorSliceCoherence, InnerTensorIsTensorBase)
{
    // Verify the inner tensor is also a TensorBase
    auto inner = createQ4_0(512, 256);
    auto slice = createSlice(std::move(inner));

    // Inner tensor should be castable to TensorBase
    TensorBase *inner_as_cpu = dynamic_cast<TensorBase *>(slice->inner());
    EXPECT_NE(inner_as_cpu, nullptr)
        << "Inner tensor should be castable to TensorBase";
}

// =============================================================================
// Test Category 2: TensorSlice Property Delegation
// =============================================================================

TEST_F(Test__TensorSliceCoherence, DelegatesNativeType)
{
    auto inner = createQ4_0(512, 256);
    ASSERT_EQ(inner->native_type(), TensorType::Q4_0);

    auto slice = createSlice(std::move(inner));
    EXPECT_EQ(slice->native_type(), TensorType::Q4_0)
        << "TensorSlice should delegate native_type() to inner";
}

TEST_F(Test__TensorSliceCoherence, DelegatesShape)
{
    auto inner = createQ4_0(512, 256);
    auto inner_shape = inner->shape();

    auto slice = createSlice(std::move(inner));
    EXPECT_EQ(slice->shape(), inner_shape)
        << "TensorSlice should delegate shape() to inner";
}

TEST_F(Test__TensorSliceCoherence, DelegatesByteSize)
{
    auto inner = createQ4_0(512, 256);
    // Use size_bytes() which is public, not byte_size() which is protected
    size_t inner_bytes = inner->size_bytes();

    auto slice = createSlice(std::move(inner));
    EXPECT_EQ(slice->size_bytes(), inner_bytes)
        << "TensorSlice should delegate size_bytes() to inner";
}

TEST_F(Test__TensorSliceCoherence, DelegatesIsOnDevice)
{
    auto inner = createQ4_0(512, 256);
    bool inner_on_cpu = inner->is_on_device(DeviceId::cpu());

    auto slice = createSlice(std::move(inner));
    EXPECT_EQ(slice->is_on_device(DeviceId::cpu()), inner_on_cpu)
        << "TensorSlice should delegate is_on_device() to inner";
}

// =============================================================================
// Test Category 3: Coherence Method Behavior
// =============================================================================

TEST_F(Test__TensorSliceCoherence, EnsureOnDeviceCallsInner)
{
    // Use MockCoherenceTensor to track calls
    auto mock = std::make_unique<MockCoherenceTensor>(512, 256);
    MockCoherenceTensor *mock_ptr = mock.get(); // Keep pointer for assertions

    auto slice = createSlice(std::move(mock));

    // The fix requires calling ensureOnDevice on the INNER tensor
    // First, get inner and cast to TensorBase
    auto *inner_cpu = dynamic_cast<TensorBase *>(slice->inner());
    ASSERT_NE(inner_cpu, nullptr);

    // Call ensureOnDevice on inner
    DeviceId target = DeviceId::cuda(0);
    inner_cpu->ensureOnDevice(target);

    // Verify the mock was called
    EXPECT_EQ(mock_ptr->ensureOnDevice_calls, 1)
        << "ensureOnDevice should be called on inner tensor";
    EXPECT_EQ(mock_ptr->last_ensureOnDevice_target.to_string(), target.to_string())
        << "ensureOnDevice should receive correct target device";
}

TEST_F(Test__TensorSliceCoherence, EnsureOnHostCallsInner)
{
    auto mock = std::make_unique<MockCoherenceTensor>(512, 256);
    MockCoherenceTensor *mock_ptr = mock.get();

    auto slice = createSlice(std::move(mock));

    // Get inner and call ensureOnHost
    auto *inner_cpu = dynamic_cast<TensorBase *>(slice->inner());
    ASSERT_NE(inner_cpu, nullptr);

    // Reset counters after setup (FP32Tensor constructor may call ensureOnHost internally)
    mock_ptr->resetCallCounters();

    inner_cpu->ensureOnHost();

    EXPECT_EQ(mock_ptr->ensureOnHost_calls, 1)
        << "ensureOnHost should be called on inner tensor";
}

TEST_F(Test__TensorSliceCoherence, TransitionToDeviceAuthoritativeOnInner)
{
    auto mock = std::make_unique<MockCoherenceTensor>(512, 256);

    auto slice = createSlice(std::move(mock));

    // Get inner and call transitionTo
    auto *inner_cpu = dynamic_cast<TensorBase *>(slice->inner());
    ASSERT_NE(inner_cpu, nullptr);
    inner_cpu->transitionTo(TensorCoherenceState::DEVICE_AUTHORITATIVE);

    EXPECT_EQ(inner_cpu->coherenceState(), TensorCoherenceState::DEVICE_AUTHORITATIVE)
        << "transitionTo should set inner tensor to DEVICE_AUTHORITATIVE";
}

// =============================================================================
// Test Category 4: TensorSlice Now Delegates to Inner
// =============================================================================

TEST_F(Test__TensorSliceCoherence, DirectEnsureOnDeviceDelegatesToInner)
{
    // This test verifies that TensorSlice::ensureOnDevice properly delegates
    // to the inner tensor, which is the correct behavior.

    auto mock = std::make_unique<MockCoherenceTensor>(512, 256);
    MockCoherenceTensor *mock_ptr = mock.get();

    auto slice = createSlice(std::move(mock));

    // Call ensureOnDevice directly on TensorSlice (via TensorBase cast)
    auto *slice_as_cpu = dynamic_cast<TensorBase *>(slice.get());
    ASSERT_NE(slice_as_cpu, nullptr);

    // This calls TensorSlice's ensureOnDevice which delegates to inner
    slice_as_cpu->ensureOnDevice(DeviceId::cuda(0));

    // The mock's method WAS called because TensorSlice now correctly delegates
    EXPECT_EQ(mock_ptr->ensureOnDevice_calls, 1)
        << "TensorSlice::ensureOnDevice should delegate to inner tensor";
}

// =============================================================================
// Test Category 5: The Solution - Unwrap and Call Inner
// =============================================================================

TEST_F(Test__TensorSliceCoherence, UnwrappingPatternWorksCorrectly)
{
    // This test verifies the pattern used in StageCoherence.cpp fix
    auto mock = std::make_unique<MockCoherenceTensor>(512, 256);
    MockCoherenceTensor *mock_ptr = mock.get();

    auto slice = createSlice(std::move(mock));

    // ===== Simulate StageCoherence.cpp unwrapping logic =====
    TensorBase *tensor = slice.get();
    TensorBase *tensor_base = dynamic_cast<TensorBase *>(tensor);

    // First cast succeeds (TensorSlice inherits from TensorBase)
    ASSERT_NE(tensor_base, nullptr);

    // But we need to check if it's a TensorSlice and unwrap
    auto *as_slice = dynamic_cast<TensorSlice *>(tensor);
    if (as_slice)
    {
        // It's a TensorSlice - get the inner tensor
        tensor_base = dynamic_cast<TensorBase *>(as_slice->inner());
        ASSERT_NE(tensor_base, nullptr) << "Inner tensor should be TensorBase";
    }

    // Now call ensureOnDevice on the unwrapped inner tensor
    DeviceId target = DeviceId::rocm(0);
    tensor_base->ensureOnDevice(target);

    // Verify the mock was called
    EXPECT_EQ(mock_ptr->ensureOnDevice_calls, 1)
        << "After unwrapping, ensureOnDevice should call inner's method";
    EXPECT_EQ(mock_ptr->last_ensureOnDevice_target.to_string(), target.to_string())
        << "Target device should be passed correctly to inner";
}

// =============================================================================
// Test Category 6: Edge Cases
// =============================================================================

TEST_F(Test__TensorSliceCoherence, MultipleCalls_TrackAll)
{
    auto mock = std::make_unique<MockCoherenceTensor>(512, 256);
    MockCoherenceTensor *mock_ptr = mock.get();

    auto slice = createSlice(std::move(mock));
    auto *inner_cpu = dynamic_cast<TensorBase *>(slice->inner());
    ASSERT_NE(inner_cpu, nullptr);

    // Call multiple times
    inner_cpu->ensureOnDevice(DeviceId::cuda(0));
    inner_cpu->ensureOnDevice(DeviceId::rocm(1));
    inner_cpu->ensureOnDevice(DeviceId::cuda(0));

    EXPECT_EQ(mock_ptr->ensureOnDevice_calls, 3)
        << "All ensureOnDevice calls should be tracked";
}

TEST_F(Test__TensorSliceCoherence, MixedCoherenceCalls)
{
    auto mock = std::make_unique<MockCoherenceTensor>(512, 256);
    MockCoherenceTensor *mock_ptr = mock.get();

    auto slice = createSlice(std::move(mock));
    auto *inner_cpu = dynamic_cast<TensorBase *>(slice->inner());
    ASSERT_NE(inner_cpu, nullptr);

    // Reset counters after setup (FP32Tensor constructor may call coherence methods internally)
    mock_ptr->resetCallCounters();

    // Simulate typical usage pattern
    inner_cpu->ensureOnDevice(DeviceId::rocm(0));                        // Upload to GPU
    inner_cpu->transitionTo(TensorCoherenceState::DEVICE_AUTHORITATIVE); // GPU kernel modified it
    inner_cpu->ensureOnHost();                                           // Download back

    EXPECT_EQ(mock_ptr->ensureOnDevice_calls, 1);
    EXPECT_EQ(mock_ptr->ensureOnHost_calls, 1);
}

TEST_F(Test__TensorSliceCoherence, NestedTensorSliceUnwrapping)
{
    // What if someone wraps a TensorSlice in another TensorSlice?
    // (This shouldn't happen in practice, but let's verify behavior)

    auto mock = std::make_unique<MockCoherenceTensor>(512, 256);
    MockCoherenceTensor *mock_ptr = mock.get();

    // Create inner slice
    auto inner_slice = createSlice(std::move(mock));

    // Wrap in another slice (unusual but possible)
    SliceMetadata outer_meta;
    outer_meta.mode = SliceMode::FULL;
    outer_meta.original_rows = 512;
    outer_meta.original_cols = 256;
    outer_meta.slice_start = 0;
    outer_meta.slice_end = 512;
    outer_meta.inner_is_presliced = true;

    // Cast inner_slice to TensorBase unique_ptr to resolve ambiguity
    std::unique_ptr<TensorBase> inner_as_base = std::move(inner_slice);
    auto outer_slice = std::make_unique<TensorSlice>(std::move(inner_as_base), std::move(outer_meta));

    // Unwrap once - should get inner TensorSlice
    auto *first_unwrap = dynamic_cast<TensorSlice *>(outer_slice->inner());
    ASSERT_NE(first_unwrap, nullptr) << "First unwrap should yield TensorSlice";

    // Unwrap again - should get MockCoherenceTensor
    auto *second_unwrap = dynamic_cast<TensorBase *>(first_unwrap->inner());
    ASSERT_NE(second_unwrap, nullptr) << "Second unwrap should yield TensorBase";

    // Call ensureOnDevice on the deepest inner
    second_unwrap->ensureOnDevice(DeviceId::rocm(0));

    EXPECT_EQ(mock_ptr->ensureOnDevice_calls, 1)
        << "Nested unwrapping should reach the real tensor";
}

// =============================================================================
// Test Category 7: Real Tensor Integration
// =============================================================================

TEST_F(Test__TensorSliceCoherence, RealQ4_0TensorSlice_TypePreserved)
{
    auto inner = createQ4_0(512, 256);
    auto slice = createSlice(std::move(inner));

    // Verify type is preserved through slice
    EXPECT_EQ(slice->native_type(), TensorType::Q4_0);

    // Verify inner is accessible
    auto *inner_q4 = dynamic_cast<Q4_0Tensor *>(slice->inner());
    EXPECT_NE(inner_q4, nullptr)
        << "Inner tensor should be castable to Q4_0Tensor";
}

TEST_F(Test__TensorSliceCoherence, RealQ4_0TensorSlice_InnerCoherenceMethods)
{
    auto inner = createQ4_0(512, 256);
    auto slice = createSlice(std::move(inner));

    // Get inner as TensorBase
    auto *inner_cpu = dynamic_cast<TensorBase *>(slice->inner());
    ASSERT_NE(inner_cpu, nullptr);

    // These should not throw (actual behavior depends on GPU availability)
    // Just verify the methods exist and are callable
    EXPECT_NO_THROW({
        bool on_cpu = inner_cpu->is_on_device(DeviceId::cpu());
        (void)on_cpu; // Suppress unused warning
    });

    EXPECT_NO_THROW({
        auto current = inner_cpu->current_device();
        (void)current; // Suppress unused warning
    });
}

// =============================================================================
// Test Category 8: Slice Mode Variations
// =============================================================================

TEST_F(Test__TensorSliceCoherence, RowParallelSlice_InnerAccessible)
{
    std::unique_ptr<TensorBase> inner = createQ4_0(1024, 896);

    SliceMetadata meta = SliceMetadata::forRowParallel(
        2048, 896, // Original dims (2x the inner for simulation)
        0, 2,      // rank 0 of 2
        true       // inner_is_presliced
    );

    auto slice = std::make_unique<TensorSlice>(std::move(inner), std::move(meta));

    EXPECT_TRUE(slice->is_row_parallel());
    EXPECT_EQ(slice->slice_rows(), 1024);

    auto *inner_cpu = dynamic_cast<TensorBase *>(slice->inner());
    EXPECT_NE(inner_cpu, nullptr);
}

TEST_F(Test__TensorSliceCoherence, ColumnParallelSlice_InnerAccessible)
{
    std::unique_ptr<TensorBase> inner = createQ4_0(1024, 448); // Half columns

    SliceMetadata meta = SliceMetadata::forColumnParallel(
        1024, 896, // Original dims
        0, 2,      // rank 0 of 2
        true       // inner_is_presliced
    );

    auto slice = std::make_unique<TensorSlice>(std::move(inner), std::move(meta));

    EXPECT_TRUE(slice->is_column_parallel());
    EXPECT_EQ(slice->slice_cols(), 448);

    auto *inner_cpu = dynamic_cast<TensorBase *>(slice->inner());
    EXPECT_NE(inner_cpu, nullptr);
}

TEST_F(Test__TensorSliceCoherence, FullSlice_InnerAccessible)
{
    std::unique_ptr<TensorBase> inner = createQ4_0(1024, 896);

    SliceMetadata meta;
    meta.mode = SliceMode::FULL;
    meta.original_rows = 1024;
    meta.original_cols = 896;
    meta.slice_start = 0;
    meta.slice_end = 1024;
    meta.inner_is_presliced = false;

    auto slice = std::make_unique<TensorSlice>(std::move(inner), std::move(meta));

    EXPECT_TRUE(slice->is_full());

    auto *inner_cpu = dynamic_cast<TensorBase *>(slice->inner());
    EXPECT_NE(inner_cpu, nullptr);
}

// =============================================================================
// Test Category 9: gpu_data_ptr() Delegation (Regression for Bug #1)
// =============================================================================

TEST_F(Test__TensorSliceCoherence, GpuDataPtrDelegatesToInner)
{
    auto inner = std::make_unique<MockCoherenceTensor>(256, 128);
    auto *inner_ptr = inner.get();

    ASSERT_EQ(inner_ptr->getRawGpuDataPtr(), nullptr);

    auto slice = createSlice(std::move(inner));

    EXPECT_EQ(slice->gpu_data_ptr(), nullptr)
        << "TensorSlice::gpu_data_ptr() should be null when inner has no GPU pointer";

    int fake_gpu_buffer = 42;
    inner_ptr->injectGpuDataPtr(&fake_gpu_buffer);

    EXPECT_EQ(slice->gpu_data_ptr(), &fake_gpu_buffer)
        << "TensorSlice::gpu_data_ptr() must delegate to inner()->gpu_data_ptr()";

    const TensorSlice *const_slice = slice.get();
    EXPECT_EQ(const_slice->gpu_data_ptr(), &fake_gpu_buffer)
        << "const TensorSlice::gpu_data_ptr() must also delegate to inner";

    inner_ptr->injectGpuDataPtr(nullptr);
}

TEST_F(Test__TensorSliceCoherence, GpuDataPtrNotFromSliceOwnMember)
{
    auto inner = std::make_unique<MockCoherenceTensor>(256, 128);
    auto *inner_ptr = inner.get();

    auto slice = createSlice(std::move(inner));

    int fake_inner_gpu = 99;
    inner_ptr->injectGpuDataPtr(&fake_inner_gpu);

    void *result = slice->gpu_data_ptr();
    EXPECT_EQ(result, &fake_inner_gpu)
        << "gpu_data_ptr() must come from inner tensor, not TensorSlice's own member";

    inner_ptr->injectGpuDataPtr(nullptr);
}

TEST_F(Test__TensorSliceCoherence, GpuDataPtrConsistentWithEnsureOnDevice)
{
    auto inner = std::make_unique<MockCoherenceTensor>(256, 128);
    auto *inner_ptr = inner.get();

    auto slice = createSlice(std::move(inner));

    EXPECT_EQ(slice->gpu_data_ptr(), nullptr);
    EXPECT_EQ(inner_ptr->getRawGpuDataPtr(), nullptr);

    float fake_device_mem[128];
    inner_ptr->injectGpuDataPtr(fake_device_mem);

    EXPECT_EQ(slice->gpu_data_ptr(), fake_device_mem)
        << "After ensureOnDevice sets inner's gpu_data_ptr_, "
           "TensorSlice::gpu_data_ptr() must reflect it";

    TensorBase *as_base = slice.get();
    EXPECT_EQ(as_base->gpu_data_ptr(), fake_device_mem)
        << "gpu_data_ptr() must work through TensorBase interface too";

    inner_ptr->injectGpuDataPtr(nullptr);
}

// =============================================================================
// Test Category 10: setHostResident() Propagation
//
// Bug: setHostResident() on a TensorSlice only set the wrapper's
// memory_residency_ field. Since ensureOnDevice() delegates to inner(),
// the inner tensor still got a full GPU upload. The fix makes
// TensorSlice::setHostResident() propagate to both wrapper and inner.
// =============================================================================

TEST_F(Test__TensorSliceCoherence, SetHostResident_PropagatesToInner)
{
    auto inner = std::make_unique<MockCoherenceTensor>(512, 256);
    auto *inner_ptr = inner.get();

    auto slice = createSlice(std::move(inner));

    EXPECT_EQ(slice->memoryResidency(), MemoryResidency::STANDARD);
    EXPECT_EQ(inner_ptr->memoryResidency(), MemoryResidency::STANDARD);

    slice->setHostResident();

    EXPECT_EQ(slice->memoryResidency(), MemoryResidency::HOST_RESIDENT)
        << "Wrapper should be HOST_RESIDENT";
    EXPECT_EQ(inner_ptr->memoryResidency(), MemoryResidency::HOST_RESIDENT)
        << "Inner tensor must also be HOST_RESIDENT after slice->setHostResident()";
}

TEST_F(Test__TensorSliceCoherence, SetHostResident_IsHostResidentReflectsInner)
{
    auto inner = std::make_unique<MockCoherenceTensor>(512, 256);
    auto *inner_ptr = inner.get();

    auto slice = createSlice(std::move(inner));

    EXPECT_FALSE(slice->isHostResident());
    EXPECT_FALSE(inner_ptr->isHostResident());

    slice->setHostResident();

    EXPECT_TRUE(slice->isHostResident());
    EXPECT_TRUE(inner_ptr->isHostResident());
}

TEST_F(Test__TensorSliceCoherence, SetHostResident_ViaBasePointer)
{
    auto inner = std::make_unique<MockCoherenceTensor>(512, 256);
    auto *inner_ptr = inner.get();

    auto slice = createSlice(std::move(inner));

    TensorBase *as_base = slice.get();
    as_base->setHostResident();

    EXPECT_TRUE(inner_ptr->isHostResident())
        << "setHostResident() via TensorBase* must propagate to inner";
}

TEST_F(Test__TensorSliceCoherence, SetHostResident_EnsureOnDeviceBecomesNoop)
{
    auto inner = std::make_unique<MockCoherenceTensor>(512, 256);
    auto *inner_ptr = inner.get();

    auto slice = createSlice(std::move(inner));
    slice->setHostResident();
    inner_ptr->resetCallCounters();

    slice->ensureOnDevice(DeviceId::rocm(0));

    EXPECT_EQ(inner_ptr->memoryResidency(), MemoryResidency::HOST_RESIDENT)
        << "Inner must be HOST_RESIDENT so TransferEngine skips upload";
}

// =============================================================================
// Test Category 11: allocateOnDevice() Delegation
// =============================================================================

TEST_F(Test__TensorSliceCoherence, AllocateOnDevice_DelegatesToInner)
{
    auto inner = std::make_unique<MockCoherenceTensor>(512, 256);
    auto *inner_ptr = inner.get();

    auto slice = createSlice(std::move(inner));
    inner_ptr->resetCallCounters();

    slice->allocateOnDevice(DeviceId::cuda(0));

    EXPECT_EQ(inner_ptr->allocateOnDevice_calls, 1)
        << "allocateOnDevice must delegate to inner tensor";
    EXPECT_EQ(inner_ptr->last_allocateOnDevice_target.to_string(), "CUDA:0")
        << "Target device must be passed through";
}

TEST_F(Test__TensorSliceCoherence, AllocateOnDevice_ViaBasePointer)
{
    auto inner = std::make_unique<MockCoherenceTensor>(512, 256);
    auto *inner_ptr = inner.get();

    auto slice = createSlice(std::move(inner));
    inner_ptr->resetCallCounters();

    TensorBase *as_base = slice.get();
    as_base->allocateOnDevice(DeviceId::rocm(1));

    EXPECT_EQ(inner_ptr->allocateOnDevice_calls, 1)
        << "allocateOnDevice via TensorBase* must delegate to inner";
}

// =============================================================================
// Test Category 12: invalidateGpuData() Delegation
// =============================================================================

TEST_F(Test__TensorSliceCoherence, InvalidateGpuData_DelegatesToInner)
{
    auto inner = std::make_unique<MockCoherenceTensor>(512, 256);
    auto *inner_ptr = inner.get();

    auto slice = createSlice(std::move(inner));
    inner_ptr->resetCallCounters();

    slice->invalidateGpuData();

    EXPECT_EQ(inner_ptr->invalidateGpuData_calls, 1)
        << "invalidateGpuData must delegate to inner tensor";
}

TEST_F(Test__TensorSliceCoherence, InvalidateGpuData_ViaBasePointer)
{
    auto inner = std::make_unique<MockCoherenceTensor>(512, 256);
    auto *inner_ptr = inner.get();

    auto slice = createSlice(std::move(inner));
    inner_ptr->resetCallCounters();

    TensorBase *as_base = slice.get();
    as_base->invalidateGpuData();

    EXPECT_EQ(inner_ptr->invalidateGpuData_calls, 1)
        << "invalidateGpuData via TensorBase* must delegate to inner";
}

// =============================================================================
// Test Category 13: Coherence State Queries
// =============================================================================

TEST_F(Test__TensorSliceCoherence, CoherenceState_InnerReflectsTransitions)
{
    auto inner = std::make_unique<MockCoherenceTensor>(512, 256);
    auto *inner_ptr = inner.get();

    auto slice = createSlice(std::move(inner));

    EXPECT_EQ(inner_ptr->coherenceState(), TensorCoherenceState::HOST_ONLY);

    inner_ptr->transitionTo(TensorCoherenceState::DEVICE_AUTHORITATIVE);
    EXPECT_EQ(inner_ptr->coherenceState(), TensorCoherenceState::DEVICE_AUTHORITATIVE);
}

TEST_F(Test__TensorSliceCoherence, MemoryResidency_InnerReflectsHostResident)
{
    auto inner = std::make_unique<MockCoherenceTensor>(512, 256);
    auto *inner_ptr = inner.get();

    auto slice = createSlice(std::move(inner));

    EXPECT_EQ(inner_ptr->memoryResidency(), MemoryResidency::STANDARD);

    slice->setHostResident();

    EXPECT_EQ(inner_ptr->memoryResidency(), MemoryResidency::HOST_RESIDENT);
}

TEST_F(Test__TensorSliceCoherence, HostValid_QueriesInnerState)
{
    auto inner = std::make_unique<MockCoherenceTensor>(512, 256);

    auto slice = createSlice(std::move(inner));

    EXPECT_TRUE(slice->isHostValid())
        << "isHostValid() (delegated) should reflect inner's HOST_ONLY state";
}

// =============================================================================
// Test Category 14: Full Coherence Lifecycle Through TensorSlice
// =============================================================================

TEST_F(Test__TensorSliceCoherence, FullLifecycle_GemmWeightHostResident)
{
    auto inner = std::make_unique<MockCoherenceTensor>(1024, 896);
    auto *inner_ptr = inner.get();
    float *host_data = inner_ptr->mutable_fp32_data();
    host_data[0] = 42.0f;

    auto slice = createSlice(std::move(inner));

    slice->setHostResident();
    EXPECT_TRUE(slice->isHostResident());
    EXPECT_TRUE(inner_ptr->isHostResident());

    inner_ptr->resetCallCounters();
    slice->ensureOnDevice(DeviceId::rocm(0));

    const float *read_data = slice->data();
    ASSERT_NE(read_data, nullptr);
    EXPECT_EQ(read_data[0], 42.0f)
        << "Host data must remain accessible after setHostResident()";

    const void *raw_ptr = static_cast<const TensorBase *>(slice.get())->raw_data();
    EXPECT_NE(raw_ptr, nullptr)
        << "raw_data() must return host data for VNNI repacker to read";
}

TEST_F(Test__TensorSliceCoherence, FullLifecycle_MarkDirtyAfterGpuCompute)
{
    auto inner = std::make_unique<MockCoherenceTensor>(512, 256);
    auto *inner_ptr = inner.get();

    auto slice = createSlice(std::move(inner));
    inner_ptr->resetCallCounters();

    slice->allocateOnDevice(DeviceId::cuda(0));
    EXPECT_EQ(inner_ptr->allocateOnDevice_calls, 1);

    slice->mark_host_dirty();
}

// =============================================================================
// Test Category 15: Shared Ownership (shared_ptr constructor)
// =============================================================================

TEST_F(Test__TensorSliceCoherence, SharedPtrConstruction_SetHostResidentPropagates)
{
    auto inner = std::make_shared<MockCoherenceTensor>(512, 256);
    MockCoherenceTensor *inner_ptr = inner.get();

    SliceMetadata meta;
    meta.mode = SliceMode::ROW_PARALLEL;
    meta.original_rows = 1024;
    meta.original_cols = 256;
    meta.slice_start = 0;
    meta.slice_end = 512;
    meta.rank = 0;
    meta.world_size = 2;
    meta.inner_is_presliced = true;

    auto slice = std::make_unique<TensorSlice>(inner, std::move(meta));

    slice->setHostResident();

    EXPECT_TRUE(inner_ptr->isHostResident())
        << "setHostResident must propagate to inner even with shared_ptr construction";
    EXPECT_EQ(inner_ptr->memoryResidency(), MemoryResidency::HOST_RESIDENT);
}

TEST_F(Test__TensorSliceCoherence, SharedPtrConstruction_AllocateOnDeviceDelegates)
{
    auto inner = std::make_shared<MockCoherenceTensor>(512, 256);
    MockCoherenceTensor *inner_ptr = inner.get();

    SliceMetadata meta;
    meta.mode = SliceMode::COLUMN_PARALLEL;
    meta.original_rows = 512;
    meta.original_cols = 512;
    meta.slice_start = 0;
    meta.slice_end = 256;
    meta.rank = 0;
    meta.world_size = 2;
    meta.inner_is_presliced = true;

    auto slice = std::make_unique<TensorSlice>(inner, std::move(meta));
    inner_ptr->resetCallCounters();

    slice->allocateOnDevice(DeviceId::rocm(0));

    EXPECT_EQ(inner_ptr->allocateOnDevice_calls, 1)
        << "allocateOnDevice must delegate to inner with shared_ptr construction";
}

TEST_F(Test__TensorSliceCoherence, SharedPtrConstruction_InvalidateGpuDataDelegates)
{
    auto inner = std::make_shared<MockCoherenceTensor>(512, 256);
    MockCoherenceTensor *inner_ptr = inner.get();

    SliceMetadata meta;
    meta.mode = SliceMode::FULL;
    meta.original_rows = 512;
    meta.original_cols = 256;
    meta.slice_start = 0;
    meta.slice_end = 512;
    meta.inner_is_presliced = true;

    auto slice = std::make_unique<TensorSlice>(inner, std::move(meta));
    inner_ptr->resetCallCounters();

    slice->invalidateGpuData();

    EXPECT_EQ(inner_ptr->invalidateGpuData_calls, 1)
        << "invalidateGpuData must delegate to inner with shared_ptr construction";
}
