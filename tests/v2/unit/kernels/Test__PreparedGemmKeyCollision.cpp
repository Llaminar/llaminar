/**
 * @file Test__PreparedGemmKeyCollision.cpp
 * @brief Regression tests for PreparedGemmKey collision after host data release
 *
 * Root cause (fixed in this commit):
 * PreparedGemmKey used tensor->raw_data() as the primary cache key. After
 * WeightManager::finalizeForDevices() releases host weight data via
 * releaseAllHostWeightData(), TensorSlice::raw_data() returns nullptr
 * (because inner tensor's host pointer is freed). When two different tensors
 * (e.g., K and V weights in TP) have the same shape, device, and prep kind,
 * their keys collide: both hash to {nullptr, N=64, K=896, device, kind}.
 * V then reuses K's prepared GEMM engine, producing completely wrong output.
 *
 * Fix: PreparedGemmKey now includes a tensor_identity fallback field. When
 * raw_data is nullptr, the tensor object pointer is used as a discriminator,
 * ensuring distinct tensors never collide even after host data release.
 *
 * This test suite verifies:
 * 1. PreparedGemmKey equality/hashing with non-null raw_data (normal case)
 * 2. PreparedGemmKey collision prevention with null raw_data (regression)
 * 3. TensorSlice raw_data() behavior before and after release
 * 4. End-to-end: two TensorSlices with same shape get distinct prepared GEMM entries
 */

#include <gtest/gtest.h>
#include "kernels/KernelFactory.h"
#include "tensors/Tensors.h"
#include "tensors/TensorSlice.h"
#include "backends/ComputeBackend.h"
#include <memory>

using namespace llaminar::v2::kernels;
using namespace llaminar2;

class Test__PreparedGemmKeyCollision : public ::testing::Test
{
protected:
    void SetUp() override
    {
        DeviceManager::instance().initialize(-1);
        KernelFactory::clearCache();
    }

    void TearDown() override
    {
        KernelFactory::clearCache();
    }

    /// Create a minimal Q4_0 tensor with valid host data
    static std::unique_ptr<Q4_0Tensor> createQ4_0Tensor(size_t rows, size_t cols)
    {
        const size_t block_size = 32;
        const size_t bytes_per_block = sizeof(Q4_0Block); // 2 + 16 = 18 bytes
        const size_t blocks_per_row = cols / block_size;
        const size_t num_blocks = rows * blocks_per_row;
        std::vector<uint8_t> raw_data(num_blocks * bytes_per_block, 0);

        // Fill with distinguishable patterns — each tensor gets different data
        // so we can detect if the wrong GEMM engine is used
        static int counter = 0;
        uint8_t fill = static_cast<uint8_t>(++counter);
        for (size_t i = 0; i < raw_data.size(); i++)
        {
            raw_data[i] = static_cast<uint8_t>((fill + i) & 0xFF);
        }

        return std::make_unique<Q4_0Tensor>(std::vector<size_t>{rows, cols}, raw_data);
    }

    /// Create a TensorSlice wrapping a Q4_0 tensor (simulates TP column-parallel weight)
    static std::unique_ptr<TensorSlice> createColumnParallelSlice(
        std::unique_ptr<TensorBase> inner, size_t original_rows, int rank, int world_size)
    {
        auto meta = SliceMetadata::forColumnParallel(
            original_rows, inner->shape()[1],
            rank, world_size,
            /*inner_is_presliced=*/true);
        return std::make_unique<TensorSlice>(std::move(inner), meta);
    }
};

// ============================================================================
// PreparedGemmKey equality and hashing tests
// ============================================================================

TEST_F(Test__PreparedGemmKeyCollision, SameRawDataPointer_SameKey)
{
    // Two keys with the same non-null raw_data should be equal
    int dummy;
    const void *ptr = &dummy;

    KernelFactory::PreparedGemmKey a{ptr, nullptr, 64, 896, DeviceId::cpu(), 0};
    KernelFactory::PreparedGemmKey b{ptr, nullptr, 64, 896, DeviceId::cpu(), 0};

    EXPECT_TRUE(a == b);

    KernelFactory::PreparedGemmKeyHash hasher;
    EXPECT_EQ(hasher(a), hasher(b));
}

TEST_F(Test__PreparedGemmKeyCollision, DifferentRawDataPointer_DifferentKey)
{
    int dummy1, dummy2;

    KernelFactory::PreparedGemmKey a{&dummy1, nullptr, 64, 896, DeviceId::cpu(), 0};
    KernelFactory::PreparedGemmKey b{&dummy2, nullptr, 64, 896, DeviceId::cpu(), 0};

    EXPECT_FALSE(a == b);
}

TEST_F(Test__PreparedGemmKeyCollision, BothNullRawData_DifferentIdentity_DifferentKey)
{
    // KEY REGRESSION TEST: two different tensors with null raw_data
    // must NOT collide even if shape/device/kind are identical
    int id1, id2;

    KernelFactory::PreparedGemmKey a{nullptr, &id1, 64, 896, DeviceId::cpu(), 0};
    KernelFactory::PreparedGemmKey b{nullptr, &id2, 64, 896, DeviceId::cpu(), 0};

    EXPECT_FALSE(a == b) << "Keys with different tensor_identity must not be equal "
                            "(prevents K/V weight collision after host data release)";

    KernelFactory::PreparedGemmKeyHash hasher;
    EXPECT_NE(hasher(a), hasher(b))
        << "Keys with different tensor_identity should hash differently";
}

TEST_F(Test__PreparedGemmKeyCollision, BothNullRawData_SameIdentity_SameKey)
{
    // Same tensor queried twice should get the same key
    int id;

    KernelFactory::PreparedGemmKey a{nullptr, &id, 64, 896, DeviceId::cpu(), 0};
    KernelFactory::PreparedGemmKey b{nullptr, &id, 64, 896, DeviceId::cpu(), 0};

    EXPECT_TRUE(a == b);

    KernelFactory::PreparedGemmKeyHash hasher;
    EXPECT_EQ(hasher(a), hasher(b));
}

TEST_F(Test__PreparedGemmKeyCollision, NonNullRawData_IdentityIgnored)
{
    // When raw_data is non-null, tensor_identity should be nullptr
    // (views sharing the same data should still match)
    int dummy;
    int id1, id2;

    KernelFactory::PreparedGemmKey a{&dummy, nullptr, 64, 896, DeviceId::cpu(), 0};
    KernelFactory::PreparedGemmKey b{&dummy, nullptr, 64, 896, DeviceId::cpu(), 0};

    EXPECT_TRUE(a == b) << "Views with same raw_data should share cache entry";
}

TEST_F(Test__PreparedGemmKeyCollision, DifferentShape_DifferentKey)
{
    int dummy;

    KernelFactory::PreparedGemmKey a{&dummy, nullptr, 64, 896, DeviceId::cpu(), 0};
    KernelFactory::PreparedGemmKey b{&dummy, nullptr, 128, 896, DeviceId::cpu(), 0};

    EXPECT_FALSE(a == b);
}

TEST_F(Test__PreparedGemmKeyCollision, DifferentDevice_DifferentKey)
{
    int dummy;

    KernelFactory::PreparedGemmKey a{&dummy, nullptr, 64, 896, DeviceId::cpu(), 0};
    KernelFactory::PreparedGemmKey b{&dummy, nullptr, 64, 896, DeviceId::cuda(0), 0};

    EXPECT_FALSE(a == b);
}

// ============================================================================
// TensorSlice raw_data behavior tests
// ============================================================================

TEST_F(Test__PreparedGemmKeyCollision, TensorSlice_RawDataReturnsInnerPointer)
{
    auto inner = createQ4_0Tensor(64, 896);
    const void *inner_raw = inner->raw_data();
    ASSERT_NE(inner_raw, nullptr) << "Fresh Q4_0 tensor should have non-null raw_data";

    auto slice = createColumnParallelSlice(std::move(inner), 128, 0, 2);

    // TensorSlice delegates raw_data to inner
    EXPECT_EQ(slice->raw_data(), inner_raw)
        << "TensorSlice::raw_data() should return inner tensor's host data pointer";
}

TEST_F(Test__PreparedGemmKeyCollision, TensorSlice_RawDataNullAfterRelease)
{
    auto inner = createQ4_0Tensor(64, 896);
    auto slice = createColumnParallelSlice(std::move(inner), 128, 0, 2);

    ASSERT_NE(slice->raw_data(), nullptr);

    // Release host data (simulates what finalizeForDevices does)
    slice->release_raw_data();

    EXPECT_EQ(slice->raw_data(), nullptr)
        << "After release_raw_data(), TensorSlice::raw_data() should return nullptr";
    EXPECT_TRUE(slice->is_raw_data_released());
}

TEST_F(Test__PreparedGemmKeyCollision, TwoSlices_SameShape_DifferentRawData)
{
    // Before release: K and V slices have different raw_data pointers
    auto k_inner = createQ4_0Tensor(64, 896);
    auto v_inner = createQ4_0Tensor(64, 896);

    const void *k_raw = k_inner->raw_data();
    const void *v_raw = v_inner->raw_data();
    ASSERT_NE(k_raw, v_raw) << "Two independent tensors should have different data pointers";

    auto k_slice = createColumnParallelSlice(std::move(k_inner), 128, 0, 2);
    auto v_slice = createColumnParallelSlice(std::move(v_inner), 128, 0, 2);

    EXPECT_NE(k_slice->raw_data(), v_slice->raw_data());

    // After release: both return nullptr
    k_slice->release_raw_data();
    v_slice->release_raw_data();

    EXPECT_EQ(k_slice->raw_data(), nullptr);
    EXPECT_EQ(v_slice->raw_data(), nullptr);
}

// ============================================================================
// End-to-end: PreparedGemm cache with TensorSlice (CPU path)
// ============================================================================

TEST_F(Test__PreparedGemmKeyCollision, PreparedGemm_TwoSlices_DistinctCacheEntries_BeforeRelease)
{
    auto k_inner = createQ4_0Tensor(64, 896);
    auto v_inner = createQ4_0Tensor(64, 896);

    auto k_slice = createColumnParallelSlice(std::move(k_inner), 128, 0, 2);
    auto v_slice = createColumnParallelSlice(std::move(v_inner), 128, 0, 2);

    // Both should get distinct prepared GEMM handles (different raw_data)
    auto *k_handle = KernelFactory::getOrCreatePreparedGemmWeights(k_slice.get(), DeviceId::cpu());
    auto *v_handle = KernelFactory::getOrCreatePreparedGemmWeights(v_slice.get(), DeviceId::cpu());

    ASSERT_NE(k_handle, nullptr);
    ASSERT_NE(v_handle, nullptr);
    EXPECT_NE(k_handle, v_handle)
        << "K and V slices must get distinct PreparedGemmHandles (different raw_data pointers)";

    // Different GEMM engines
    auto *k_engine = KernelFactory::getOrCreateGemmEngine(k_handle);
    auto *v_engine = KernelFactory::getOrCreateGemmEngine(v_handle);
    ASSERT_NE(k_engine, nullptr);
    ASSERT_NE(v_engine, nullptr);
    EXPECT_NE(k_engine, v_engine)
        << "K and V slices must get distinct GEMM engines";
}

TEST_F(Test__PreparedGemmKeyCollision, PreparedGemm_TwoSlices_DistinctAfterRelease)
{
    // This is the KEY REGRESSION TEST for the bug:
    // After host data release, K and V must still get distinct cache entries.
    auto k_inner = createQ4_0Tensor(64, 896);
    auto v_inner = createQ4_0Tensor(64, 896);

    auto k_slice = createColumnParallelSlice(std::move(k_inner), 128, 0, 2);
    auto v_slice = createColumnParallelSlice(std::move(v_inner), 128, 0, 2);

    // Phase 1: Prepare GEMM weights (like packGemmWeights does before release)
    auto *k_handle = KernelFactory::getOrCreatePreparedGemmWeights(k_slice.get(), DeviceId::cpu());
    auto *v_handle = KernelFactory::getOrCreatePreparedGemmWeights(v_slice.get(), DeviceId::cpu());
    ASSERT_NE(k_handle, v_handle);

    auto *k_engine_before = KernelFactory::getOrCreateGemmEngine(k_handle);
    auto *v_engine_before = KernelFactory::getOrCreateGemmEngine(v_handle);
    ASSERT_NE(k_engine_before, v_engine_before);

    // Phase 2: Release host data (like releaseAllHostWeightData does)
    k_slice->release_raw_data();
    v_slice->release_raw_data();
    ASSERT_EQ(k_slice->raw_data(), nullptr);
    ASSERT_EQ(v_slice->raw_data(), nullptr);

    // Phase 3: Re-query (like FusedQKVGEMMStage does at inference time)
    // With dual-key registration: fallback keys were pre-registered during Phase 1,
    // so post-release lookups hit the cache and return the same handles.
    auto *k_handle_after = KernelFactory::getOrCreatePreparedGemmWeights(k_slice.get(), DeviceId::cpu());
    auto *v_handle_after = KernelFactory::getOrCreatePreparedGemmWeights(v_slice.get(), DeviceId::cpu());

    ASSERT_NE(k_handle_after, nullptr);
    ASSERT_NE(v_handle_after, nullptr);

    // Handles should be the same objects as before release (cache hit on fallback key)
    EXPECT_EQ(k_handle_after, k_handle)
        << "Post-release lookup should return same handle via pre-registered fallback key";
    EXPECT_EQ(v_handle_after, v_handle)
        << "Post-release lookup should return same handle via pre-registered fallback key";

    // CRITICAL: K and V must NEVER return the same handle
    EXPECT_NE(k_handle_after, v_handle_after)
        << "REGRESSION: K and V PreparedGemmHandles must be distinct even after "
           "host data release (raw_data=nullptr). Before the fix, both K and V "
           "hashed to {nullptr, 64, 896, cpu, 0} and V reused K's GEMM engine.";

    auto *k_engine_after = KernelFactory::getOrCreateGemmEngine(k_handle_after);
    auto *v_engine_after = KernelFactory::getOrCreateGemmEngine(v_handle_after);
    EXPECT_NE(k_engine_after, v_engine_after)
        << "REGRESSION: K and V GEMM engines must be distinct after host data release";
}

TEST_F(Test__PreparedGemmKeyCollision, PreparedGemm_SameTensorTwice_CacheHit)
{
    auto inner = createQ4_0Tensor(64, 896);
    auto slice = createColumnParallelSlice(std::move(inner), 128, 0, 2);

    auto *handle1 = KernelFactory::getOrCreatePreparedGemmWeights(slice.get(), DeviceId::cpu());
    auto *handle2 = KernelFactory::getOrCreatePreparedGemmWeights(slice.get(), DeviceId::cpu());

    EXPECT_EQ(handle1, handle2) << "Same tensor queried twice should return same handle (cache hit)";

    // Also after release — fallback key was pre-registered, should cache-hit
    slice->release_raw_data();
    auto *handle3 = KernelFactory::getOrCreatePreparedGemmWeights(slice.get(), DeviceId::cpu());
    EXPECT_EQ(handle1, handle3) << "After host release, fallback key should resolve to same handle";
}

TEST_F(Test__PreparedGemmKeyCollision, PreparedGemm_DifferentShapes_AlwaysDistinct)
{
    // Even with same raw_data address (hypothetical), different shapes = different keys
    auto k_inner = createQ4_0Tensor(64, 896);
    auto q_inner = createQ4_0Tensor(448, 896);

    auto k_slice = createColumnParallelSlice(std::move(k_inner), 128, 0, 2);
    // Q has different shape so use a different meta
    auto q_meta = SliceMetadata::forColumnParallel(896, 896, 0, 2, true);
    std::unique_ptr<TensorBase> q_base = std::move(q_inner);
    auto q_slice = std::make_unique<TensorSlice>(std::move(q_base), q_meta);

    auto *k_handle = KernelFactory::getOrCreatePreparedGemmWeights(k_slice.get(), DeviceId::cpu());
    auto *q_handle = KernelFactory::getOrCreatePreparedGemmWeights(q_slice.get(), DeviceId::cpu());

    EXPECT_NE(k_handle, q_handle) << "Tensors with different shapes must always get distinct handles";
}

TEST_F(Test__PreparedGemmKeyCollision, PreparedGemm_NeverPrepared_ThrowsAfterRelease)
{
    // If a tensor's host data is released without ever being prepared,
    // getOrCreatePreparedGemmWeights must throw (not silently create garbage).
    auto inner = createQ4_0Tensor(64, 896);
    auto slice = createColumnParallelSlice(std::move(inner), 128, 0, 2);

    // Skip preparation — go straight to release
    slice->release_raw_data();
    ASSERT_EQ(slice->raw_data(), nullptr);

    EXPECT_THROW(
        KernelFactory::getOrCreatePreparedGemmWeights(slice.get(), DeviceId::cpu()),
        std::runtime_error)
        << "Querying unprepared tensor after host release must throw, not create garbage GEMM";
}
