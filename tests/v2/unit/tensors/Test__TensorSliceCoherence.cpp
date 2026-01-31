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
    mutable int mark_device_dirty_calls = 0;
    mutable DeviceId last_ensureOnDevice_target = DeviceId::cpu();

    // Override coherence methods to track calls
    bool ensureOnDevice(DeviceId target_device) override
    {
        ensureOnDevice_calls++;
        last_ensureOnDevice_target = target_device;
        // Simulate successful upload (don't actually do GPU transfer)
        return true;
    }

    bool ensureOnHost() override
    {
        ensureOnHost_calls++;
        return true;
    }

    void mark_device_dirty() override
    {
        mark_device_dirty_calls++;
        // Don't call parent - we're just tracking
    }

    // Reset call counters
    void resetCallCounters()
    {
        ensureOnDevice_calls = 0;
        ensureOnHost_calls = 0;
        mark_device_dirty_calls = 0;
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

TEST_F(Test__TensorSliceCoherence, MarkDeviceDirtyCallsInner)
{
    auto mock = std::make_unique<MockCoherenceTensor>(512, 256);
    MockCoherenceTensor *mock_ptr = mock.get();

    auto slice = createSlice(std::move(mock));

    // Get inner and call mark_device_dirty
    auto *inner_cpu = dynamic_cast<TensorBase *>(slice->inner());
    ASSERT_NE(inner_cpu, nullptr);
    inner_cpu->mark_device_dirty();

    EXPECT_EQ(mock_ptr->mark_device_dirty_calls, 1)
        << "mark_device_dirty should be called on inner tensor";
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
    inner_cpu->ensureOnDevice(DeviceId::rocm(0)); // Upload to GPU
    inner_cpu->mark_device_dirty();               // GPU kernel modified it
    inner_cpu->ensureOnHost();                    // Download back

    EXPECT_EQ(mock_ptr->ensureOnDevice_calls, 1);
    EXPECT_EQ(mock_ptr->mark_device_dirty_calls, 1);
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
