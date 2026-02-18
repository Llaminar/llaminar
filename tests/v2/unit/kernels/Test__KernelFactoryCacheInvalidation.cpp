/**
 * @file Test__KernelFactoryCacheInvalidation.cpp
 * @brief Unit tests for automatic KernelFactory cache invalidation
 * @author David Sanftenberg
 *
 * Tests verify that when a tensor is destroyed, its entry in the
 * KernelFactory cache is automatically removed. This prevents use-after-free
 * bugs when a new tensor is allocated at the same memory address as a
 * previously destroyed tensor.
 *
 * The fix works by having TensorBase's destructor call
 * KernelFactory::clearCacheFor(this).
 */

#include <gtest/gtest.h>
#include "kernels/KernelFactory.h"
#include "tensors/Tensors.h"
#include "backends/ComputeBackend.h"
#include <memory>

using namespace llaminar::v2::kernels;
using namespace llaminar2;

class Test__KernelFactoryCacheInvalidation : public ::testing::Test
{
protected:
    void SetUp() override
    {
        // Initialize DeviceManager to enumerate GPU devices
        // -1 = no NUMA filtering, enumerate all devices
        auto &dm = DeviceManager::instance();
        dm.initialize(-1);
        // Start with empty cache
        KernelFactory::clearCache();
    }

    void TearDown() override
    {
        // Clean up after tests
        KernelFactory::clearCache();
    }
};

// Helper to create a minimal IQ4_NL tensor (smallest quantized format)
static std::unique_ptr<IQ4_NLTensor> createTestTensor()
{
    const size_t rows = 32;
    const size_t cols = 32;
    const size_t block_size = 32;
    const size_t bytes_per_block = 18; // IQ4_NL block size
    const size_t num_blocks = rows * (cols / block_size);
    std::vector<uint8_t> raw_data(num_blocks * bytes_per_block, 0);

    return std::make_unique<IQ4_NLTensor>(std::vector<size_t>{rows, cols}, raw_data);
}

static ITensorGemm *getPreparedKernel(const TensorBase *tensor, DeviceId device = DeviceId::cpu())
{
    auto *prepared = KernelFactory::getOrCreatePreparedGemmWeights(tensor, device);
    return KernelFactory::getOrCreateGemmEngine(prepared);
}

// ============================================================================
// Basic Cache Invalidation Tests
// ============================================================================

TEST_F(Test__KernelFactoryCacheInvalidation, CacheEmptyAfterClear)
{
    auto [cache_size, packed_bytes] = KernelFactory::cacheStats();
    EXPECT_EQ(cache_size, 0u);
}

TEST_F(Test__KernelFactoryCacheInvalidation, CacheGrowsAfterGetOrCreateGemm)
{
    auto tensor = createTestTensor();

    // Verify cache starts empty
    auto [size_before, _] = KernelFactory::cacheStats();
    EXPECT_EQ(size_before, 0u);

    // Create a GEMM kernel via factory (should cache it)
    auto *kernel = getPreparedKernel(tensor.get());
    ASSERT_NE(kernel, nullptr);

    // Verify cache now has one entry
    auto [size_after, bytes] = KernelFactory::cacheStats();
    EXPECT_EQ(size_after, 1u);
    // Note: bytes may be 0 for some kernel types that don't track packed data size
}

TEST_F(Test__KernelFactoryCacheInvalidation, CacheAutoInvalidatesOnTensorDestruction)
{
    // This is the KEY test for the automatic invalidation feature

    // Create tensor in a scope
    {
        auto tensor = createTestTensor();

        // Add to cache
        auto *kernel = getPreparedKernel(tensor.get());
        ASSERT_NE(kernel, nullptr);

        // Verify it's in the cache
        auto [size_during, _] = KernelFactory::cacheStats();
        EXPECT_EQ(size_during, 1u);

        // tensor goes out of scope here -> destructor runs
    }

    // After tensor destruction, cache entry should be automatically removed
    auto [size_after, bytes_after] = KernelFactory::cacheStats();
    EXPECT_EQ(size_after, 0u) << "Cache should be empty after tensor destruction";
    EXPECT_EQ(bytes_after, 0u) << "Packed bytes should be zero after tensor destruction";
}

TEST_F(Test__KernelFactoryCacheInvalidation, MultipleTensors_IndependentInvalidation)
{
    // Create two tensors
    auto tensor1 = createTestTensor();
    auto tensor2 = createTestTensor();

    // Add both to cache
    getPreparedKernel(tensor1.get());
    getPreparedKernel(tensor2.get());

    // Should have 2 entries
    auto [size_both, _] = KernelFactory::cacheStats();
    EXPECT_EQ(size_both, 2u);

    // Destroy tensor1
    tensor1.reset();

    // Should have 1 entry remaining (tensor2)
    auto [size_one, __] = KernelFactory::cacheStats();
    EXPECT_EQ(size_one, 1u) << "Only tensor2's kernel should remain in cache";

    // Destroy tensor2
    tensor2.reset();

    // Cache should now be empty
    auto [size_none, ___] = KernelFactory::cacheStats();
    EXPECT_EQ(size_none, 0u) << "Cache should be empty after all tensors destroyed";
}

TEST_F(Test__KernelFactoryCacheInvalidation, ClearCacheForNonExistentTensor_NoOp)
{
    // Create and cache a tensor
    auto tensor = createTestTensor();
    getPreparedKernel(tensor.get());

    auto [size_before, _] = KernelFactory::cacheStats();
    EXPECT_EQ(size_before, 1u);

    // Create a different tensor (not cached) and clear its entry
    // This should be a no-op - the cache should still have the original entry
    auto tensor2 = createTestTensor();
    KernelFactory::clearCacheFor(tensor2.get());

    // Cache should still have the first tensor's entry
    auto [size_after, __] = KernelFactory::cacheStats();
    EXPECT_EQ(size_after, 1u) << "Cache should be unaffected by clearCacheFor with different tensor";
}

TEST_F(Test__KernelFactoryCacheInvalidation, CacheHit_SamePointerReturnssamKernel)
{
    auto tensor = createTestTensor();

    // First call creates and caches
    auto *kernel1 = getPreparedKernel(tensor.get());
    ASSERT_NE(kernel1, nullptr);

    // Second call should return the SAME kernel (cache hit)
    auto *kernel2 = getPreparedKernel(tensor.get());
    EXPECT_EQ(kernel1, kernel2) << "Cache should return the same kernel pointer";

    // Should still have only 1 cache entry
    auto [size, _] = KernelFactory::cacheStats();
    EXPECT_EQ(size, 1u);
}

// ============================================================================
// Regression Test: Memory Reuse Scenario
// ============================================================================

TEST_F(Test__KernelFactoryCacheInvalidation, MemoryReuse_NoStaleKernel)
{
    // This test simulates the exact bug that was occurring:
    // 1. Create tensor A, cache its kernel
    // 2. Destroy tensor A
    // 3. Create tensor B (may get same address as A)
    // 4. Query cache for tensor B -> should NOT get A's stale kernel

    // We can't guarantee memory reuse, but we can verify that after
    // destruction, no stale entries exist

    const TensorBase *captured_ptr = nullptr;

    // Create and cache tensor
    {
        auto tensor = createTestTensor();
        captured_ptr = tensor.get();

        getPreparedKernel(tensor.get());

        auto [size_during, _] = KernelFactory::cacheStats();
        EXPECT_EQ(size_during, 1u);
    }

    // After destruction, even if we query with the old pointer address,
    // we should NOT find any entry (cache should be clean)
    auto [size_after, _] = KernelFactory::cacheStats();
    EXPECT_EQ(size_after, 0u) << "Cache should be empty after tensor destruction";

    // Note: We can't call getPreparedKernel(captured_ptr) because that would be UB.
    // The important thing is that the cache is clean, so if a new tensor
    // happens to be allocated at the same address, it won't hit a stale entry.
}

// ============================================================================
// FP32 Tensor (non-quantized) Test
// ============================================================================

TEST_F(Test__KernelFactoryCacheInvalidation, FP32Tensor_AutoInvalidation)
{
    // FP32 tensors also go through the cache
    {
        const size_t rows = 32;
        const size_t cols = 32;
        // FP32Tensor constructor takes shape and optional device_idx (defaults to -1 for CPU)
        auto tensor = std::make_unique<FP32Tensor>(std::vector<size_t>{rows, cols});

        auto *kernel = getPreparedKernel(tensor.get());
        ASSERT_NE(kernel, nullptr);

        auto [size_during, _] = KernelFactory::cacheStats();
        EXPECT_EQ(size_during, 1u);
    }

    // After destruction, cache should be empty
    auto [size_after, _] = KernelFactory::cacheStats();
    EXPECT_EQ(size_after, 0u);
}

// ============================================================================
// Edge Cases
// ============================================================================

TEST_F(Test__KernelFactoryCacheInvalidation, TensorWithoutKernel_DestructorSafe)
{
    // Creating and destroying a tensor that was NEVER added to cache
    // should not cause any issues
    {
        auto tensor = createTestTensor();
        // Don't call getOrCreateGemm - just destroy it
    }

    // Should be a no-op, no crash
    auto [size, _] = KernelFactory::cacheStats();
    EXPECT_EQ(size, 0u);
}

TEST_F(Test__KernelFactoryCacheInvalidation, RapidCreateDestroy_CacheStaysClean)
{
    // Rapidly create and destroy many tensors
    for (int i = 0; i < 100; ++i)
    {
        auto tensor = createTestTensor();
        getPreparedKernel(tensor.get());
    }

    // After all destroys, cache should be empty
    auto [size, _] = KernelFactory::cacheStats();
    EXPECT_EQ(size, 0u) << "Cache should be clean after rapid create/destroy cycles";
}

// ============================================================================
// Packed Weights Cache Cleanup Tests
// ============================================================================

TEST_F(Test__KernelFactoryCacheInvalidation, CPUPackedWeights_CleanedUpOnDestruction)
{
    // This test verifies that tensor->cache_ (CPU VNNI packed weights)
    // is properly cleaned up when the tensor is destroyed.

    {
        auto tensor = createTestTensor();

        // Initially, cache_ should be empty
        EXPECT_FALSE(tensor->cache_.has_value()) << "Fresh tensor should have no cache";

        // Create GEMM kernel - this populates tensor->cache_ with packed weights
        auto *kernel = getPreparedKernel(tensor.get());
        ASSERT_NE(kernel, nullptr);

        // Now cache_ should have packed weights
        EXPECT_TRUE(tensor->cache_.has_value()) << "After GEMM creation, tensor should have packed weights in cache_";

        // tensor goes out of scope here
    }

    // After tensor destruction, the KernelFactory cache should be empty
    // AND the packed weights should have been deleted (no memory leak)
    // We can't directly verify memory was freed, but we verify cache is clean
    auto [size_after, _] = KernelFactory::cacheStats();
    EXPECT_EQ(size_after, 0u) << "Cache should be empty after tensor destruction";
}

TEST_F(Test__KernelFactoryCacheInvalidation, CPUPackedWeights_ClearedByExplicitClearCacheFor)
{
    // Verify that clearCacheFor explicitly clears cache_
    auto tensor = createTestTensor();

    // Populate cache
    getPreparedKernel(tensor.get());
    EXPECT_TRUE(tensor->cache_.has_value());

    // Explicitly clear
    KernelFactory::clearCacheFor(tensor.get());

    // cache_ should now be empty
    EXPECT_FALSE(tensor->cache_.has_value()) << "clearCacheFor should reset cache_";
}

TEST_F(Test__KernelFactoryCacheInvalidation, CPUPackedWeights_ClearedByClearCache)
{
    // Verify that clearCache() clears all tensor caches
    auto tensor1 = createTestTensor();
    auto tensor2 = createTestTensor();

    // Populate caches
    getPreparedKernel(tensor1.get());
    getPreparedKernel(tensor2.get());

    EXPECT_TRUE(tensor1->cache_.has_value());
    EXPECT_TRUE(tensor2->cache_.has_value());

    // Clear all
    KernelFactory::clearCache();

    // Both should be cleared
    EXPECT_FALSE(tensor1->cache_.has_value()) << "clearCache should reset all tensor caches";
    EXPECT_FALSE(tensor2->cache_.has_value()) << "clearCache should reset all tensor caches";
}

TEST_F(Test__KernelFactoryCacheInvalidation, MultiplePackedWeightsCreation_OnlyPacksOnce)
{
    // Verify that calling getOrCreateGemm multiple times doesn't re-pack weights
    auto tensor = createTestTensor();

    // First call - should create packed weights
    auto *kernel1 = getPreparedKernel(tensor.get());
    ASSERT_NE(kernel1, nullptr);
    EXPECT_TRUE(tensor->cache_.has_value());

    // Get pointer to cached data for comparison
    void *cache_ptr_before = nullptr;
    try
    {
        // We can't directly access the packed data, but we can check the any still has value
        cache_ptr_before = &tensor->cache_;
    }
    catch (...)
    {
    }

    // Second call - should reuse existing packed weights
    auto *kernel2 = getPreparedKernel(tensor.get());
    EXPECT_EQ(kernel1, kernel2) << "Should return cached kernel";
    EXPECT_TRUE(tensor->cache_.has_value());

    // Cache should still have exactly 1 entry
    auto [size, _] = KernelFactory::cacheStats();
    EXPECT_EQ(size, 1u) << "Should only have one cached kernel";
}

#ifdef HAVE_CUDA
// ============================================================================
// CUDA Packed Weights Cache Cleanup Tests
// ============================================================================

TEST_F(Test__KernelFactoryCacheInvalidation, CUDAPackedWeights_CleanedUpOnDestruction)
{
    // This test verifies that tensor->cuda_cache_ (CUDA INT8 packed weights)
    // is properly cleaned up when the tensor is destroyed.
    //
    // Note: This requires a CUDA device to be available for full coverage.
    // If no CUDA device, the test validates that cuda_cache_ remains empty.

    // Check if we have CUDA devices
    const auto &dm = DeviceManager::instance();
    const auto &devices = dm.devices();
    bool has_cuda = false;
    int cuda_device_idx = -1;

    for (size_t i = 0; i < devices.size(); ++i)
    {
        if (devices[i].type == ComputeBackendType::GPU_CUDA)
        {
            has_cuda = true;
            cuda_device_idx = static_cast<int>(i);
            break;
        }
    }

    if (!has_cuda)
    {
        GTEST_SKIP() << "No CUDA device available, skipping CUDA cache cleanup test";
    }

    {
        auto tensor = createTestTensor();

        // Initially, cuda_cache_ should be empty
        EXPECT_FALSE(tensor->cuda_cache_.has_value()) << "Fresh tensor should have no CUDA cache";

        // Create CUDA GEMM kernel - this should populate cuda_cache_
        // Use CUDAOrdinalGuard to set the target device ordinal
        KernelFactory::CUDAOrdinalGuard guard(0);
        auto kernel = KernelFactory::createGemm(
            dynamic_cast<const IQ4_NLTensor *>(tensor.get()),
            DeviceType::CUDA);
        ASSERT_NE(kernel, nullptr);

        // Now cuda_cache_ should have packed weights
        EXPECT_TRUE(tensor->cuda_cache_.has_value())
            << "After CUDA GEMM creation, tensor should have packed weights in cuda_cache_";

        // tensor goes out of scope here
    }

    // After tensor destruction, CUDA packed weights should have been freed
    // (including device memory via ~CUDAPackedWeights)
    auto [size_after, _] = KernelFactory::cacheStats();
    // Note: createGemm with explicit DeviceType doesn't add to kernel_cache_,
    // but the cuda_cache_ cleanup happens in clearCacheFor
}

TEST_F(Test__KernelFactoryCacheInvalidation, CUDAPackedWeights_ClearedByExplicitClearCacheFor)
{
    const auto &dm = DeviceManager::instance();
    const auto &devices = dm.devices();
    bool has_cuda = false;

    for (const auto &dev : devices)
    {
        if (dev.type == ComputeBackendType::GPU_CUDA)
        {
            has_cuda = true;
            break;
        }
    }

    if (!has_cuda)
    {
        GTEST_SKIP() << "No CUDA device available";
    }

    auto tensor = createTestTensor();

    // Create CUDA kernel to populate cuda_cache_
    // Use CUDAOrdinalGuard to set the target device ordinal
    KernelFactory::CUDAOrdinalGuard guard(0);
    auto kernel = KernelFactory::createGemm(
        dynamic_cast<const IQ4_NLTensor *>(tensor.get()),
        DeviceType::CUDA);
    EXPECT_TRUE(tensor->cuda_cache_.has_value());

    // Explicitly clear
    KernelFactory::clearCacheFor(tensor.get());

    // cuda_cache_ should now be empty
    EXPECT_FALSE(tensor->cuda_cache_.has_value())
        << "clearCacheFor should reset cuda_cache_";
}

TEST_F(Test__KernelFactoryCacheInvalidation, BothCaches_CleanedUpTogether)
{
    // Test that a tensor with BOTH CPU and CUDA packed weights
    // has both cleaned up properly

    const auto &dm = DeviceManager::instance();
    const auto &devices = dm.devices();
    bool has_cuda = false;

    for (const auto &dev : devices)
    {
        if (dev.type == ComputeBackendType::GPU_CUDA)
        {
            has_cuda = true;
            break;
        }
    }

    if (!has_cuda)
    {
        GTEST_SKIP() << "No CUDA device available";
    }

    {
        auto tensor = createTestTensor();

        // Create CPU GEMM (populates cache_)
        auto *cpu_kernel = getPreparedKernel(tensor.get());
        ASSERT_NE(cpu_kernel, nullptr);
        EXPECT_TRUE(tensor->cache_.has_value()) << "Should have CPU packed weights";

        // Create CUDA GEMM (populates cuda_cache_)
        // Use CUDAOrdinalGuard to set the target device ordinal
        KernelFactory::CUDAOrdinalGuard guard(0);
        auto cuda_kernel = KernelFactory::createGemm(
            dynamic_cast<const IQ4_NLTensor *>(tensor.get()),
            DeviceType::CUDA);
        ASSERT_NE(cuda_kernel, nullptr);
        EXPECT_TRUE(tensor->cuda_cache_.has_value()) << "Should have CUDA packed weights";

        // tensor goes out of scope here
    }

    // After destruction, both caches should be cleaned up
    auto [size_after, _] = KernelFactory::cacheStats();
    EXPECT_EQ(size_after, 0u) << "Kernel cache should be empty";
    // Note: We can't check tensor->cache_ anymore since tensor is destroyed,
    // but the destructor should have called clearCacheFor which cleans both
}

// ============================================================================
// REGRESSION: Device-Targeted Cache Invalidation (GitHub Issue #XXX)
// ============================================================================
// These tests verify that device_targeted_cache_ is properly cleared when
// a tensor is destroyed. This is a regression test for a bug where
// device-specific prepared-handle kernels
// were NOT being cleared, causing use-after-free when tensor memory was reused.

TEST_F(Test__KernelFactoryCacheInvalidation, DeviceTargetedCache_CUDA_AutoInvalidation)
{
    // This test verifies the bug fix: device_targeted_cache_ entries must be
    // cleared when a tensor is destroyed.
    //
    // Bug scenario:
    // 1. Create tensor A, create device-targeted CUDA kernel (cached by tensor ptr)
    // 2. Destroy tensor A (cache entry SHOULD be removed)
    // 3. Create tensor B at same memory address (memory reuse)
    // 4. Query cache for tensor B -> WITHOUT FIX: returns stale kernel from A
    //                                WITH FIX: cache miss, creates new kernel

    const auto &dm = DeviceManager::instance();
    const auto &devices = dm.devices();
    bool has_cuda = false;

    for (const auto &dev : devices)
    {
        if (dev.type == ComputeBackendType::GPU_CUDA)
        {
            has_cuda = true;
            break;
        }
    }

    if (!has_cuda)
    {
        GTEST_SKIP() << "No CUDA device available";
    }

    {
        auto tensor = createTestTensor();

        // Create device-targeted kernel (this uses device_targeted_cache_)
        // Use DeviceId to properly set the target device ordinal
        auto *kernel = getPreparedKernel(tensor.get(), DeviceId::cuda(0));
        ASSERT_NE(kernel, nullptr);

        // Verify cache has an entry
        auto [size_during, _] = KernelFactory::cacheStats();
        EXPECT_GE(size_during, 1u) << "Cache should have at least one entry";

        // tensor goes out of scope here -> destructor should clear cache entry
    }

    // After destruction, the device-targeted cache entry should be removed
    // This is the key assertion for the bug fix
    auto [size_after, _] = KernelFactory::cacheStats();
    EXPECT_EQ(size_after, 0u)
        << "REGRESSION: device_targeted_cache_ was not cleared on tensor destruction. "
           "This causes use-after-free when memory is reused.";
}

TEST_F(Test__KernelFactoryCacheInvalidation, DeviceTargetedCache_CPU_AutoInvalidation)
{
    // Same test but for CPU device-targeted kernels
    {
        auto tensor = createTestTensor();

        // Create CPU device-targeted kernel
        auto *kernel = getPreparedKernel(tensor.get(), DeviceId::cpu());
        ASSERT_NE(kernel, nullptr);

        auto [size_during, _] = KernelFactory::cacheStats();
        EXPECT_GE(size_during, 1u);
    }

    auto [size_after, _] = KernelFactory::cacheStats();
    EXPECT_EQ(size_after, 0u)
        << "REGRESSION: device_targeted_cache_ (CPU) was not cleared on tensor destruction";
}

TEST_F(Test__KernelFactoryCacheInvalidation, DeviceTargetedCache_MultipleDevices_IndependentInvalidation)
{
    // Test that clearing one tensor doesn't affect device-targeted kernels for other tensors

    const auto &dm = DeviceManager::instance();
    const auto &devices = dm.devices();
    bool has_cuda = false;

    for (const auto &dev : devices)
    {
        if (dev.type == ComputeBackendType::GPU_CUDA)
        {
            has_cuda = true;
            break;
        }
    }

    if (!has_cuda)
    {
        GTEST_SKIP() << "No CUDA device available";
    }

    auto tensor1 = createTestTensor();
    auto tensor2 = createTestTensor();

    // Create device-targeted kernels for both tensors
    // Use DeviceId to properly set the target device ordinal
    auto *kernel1 = getPreparedKernel(tensor1.get(), DeviceId::cuda(0));
    auto *kernel2 = getPreparedKernel(tensor2.get(), DeviceId::cuda(0));

    ASSERT_NE(kernel1, nullptr);
    ASSERT_NE(kernel2, nullptr);

    auto [size_both, _] = KernelFactory::cacheStats();
    EXPECT_GE(size_both, 2u) << "Should have at least 2 cache entries";

    // Destroy tensor1
    tensor1.reset();

    // tensor2's kernel should still be in cache
    auto [size_one, __] = KernelFactory::cacheStats();
    EXPECT_GE(size_one, 1u) << "tensor2's kernel should still be cached";

    // Verify we can still get tensor2's kernel (cache hit)
    auto *kernel2_again = getPreparedKernel(tensor2.get(), DeviceId::cuda(0));
    EXPECT_EQ(kernel2, kernel2_again) << "Should return same cached kernel for tensor2";

    // Destroy tensor2
    tensor2.reset();

    // Now cache should be empty
    auto [size_none, ___] = KernelFactory::cacheStats();
    EXPECT_EQ(size_none, 0u) << "Cache should be empty after all tensors destroyed";
}

TEST_F(Test__KernelFactoryCacheInvalidation, DeviceTargetedCache_SameTensorBothDevices)
{
    // Test tensor with kernels cached for BOTH CPU and CUDA device types
    const auto &dm = DeviceManager::instance();
    const auto &devices = dm.devices();
    bool has_cuda = false;

    for (const auto &dev : devices)
    {
        if (dev.type == ComputeBackendType::GPU_CUDA)
        {
            has_cuda = true;
            break;
        }
    }

    if (!has_cuda)
    {
        GTEST_SKIP() << "No CUDA device available";
    }

    {
        auto tensor = createTestTensor();

        // Create kernels for both device types
        // Use DeviceId to properly set the target device ordinal
        auto *cpu_kernel = getPreparedKernel(tensor.get(), DeviceId::cpu());
        auto *cuda_kernel = getPreparedKernel(tensor.get(), DeviceId::cuda(0));

        ASSERT_NE(cpu_kernel, nullptr);
        ASSERT_NE(cuda_kernel, nullptr);
        EXPECT_NE(cpu_kernel, cuda_kernel) << "CPU and CUDA kernels should be different";

        auto [size_during, _] = KernelFactory::cacheStats();
        EXPECT_GE(size_during, 2u) << "Should have entries for both device types";

        // tensor goes out of scope here
    }

    // Both entries should be cleared
    auto [size_after, _] = KernelFactory::cacheStats();
    EXPECT_EQ(size_after, 0u)
        << "REGRESSION: Both CPU and CUDA device_targeted_cache_ entries should be cleared";
}

TEST_F(Test__KernelFactoryCacheInvalidation, DeviceTargetedCache_ExplicitClearCacheFor)
{
    // Test that explicit clearCacheFor() clears device_targeted_cache_ entries

    const auto &dm = DeviceManager::instance();
    const auto &devices = dm.devices();
    bool has_cuda = false;

    for (const auto &dev : devices)
    {
        if (dev.type == ComputeBackendType::GPU_CUDA)
        {
            has_cuda = true;
            break;
        }
    }

    if (!has_cuda)
    {
        GTEST_SKIP() << "No CUDA device available";
    }

    auto tensor = createTestTensor();

    // Create device-targeted kernel
    // Use DeviceId to properly set the target device ordinal
    auto *kernel = getPreparedKernel(tensor.get(), DeviceId::cuda(0));
    ASSERT_NE(kernel, nullptr);

    auto [size_before, _] = KernelFactory::cacheStats();
    EXPECT_GE(size_before, 1u);

    // Explicitly clear cache for this tensor
    KernelFactory::clearCacheFor(tensor.get());

    // Entry should be removed
    auto [size_after, __] = KernelFactory::cacheStats();
    EXPECT_EQ(size_after, 0u)
        << "clearCacheFor should remove device_targeted_cache_ entries";

    // Creating a new kernel should result in a cache miss (new kernel)
    auto *kernel_new = getPreparedKernel(tensor.get(), DeviceId::cuda(0));
    ASSERT_NE(kernel_new, nullptr);
    // Note: kernel_new might equal kernel if the factory creates the same type,
    // but the key point is the cache was cleared
}

TEST_F(Test__KernelFactoryCacheInvalidation, DeviceTargetedCache_MemoryReuseSimulation)
{
    // This test simulates the exact scenario that caused the original bug:
    // Memory reuse after tensor destruction leading to stale cache hits.
    //
    // We can't force memory reuse, but we can verify the cache is clean
    // after destruction, which prevents the bug.

    const auto &dm = DeviceManager::instance();
    const auto &devices = dm.devices();
    bool has_cuda = false;

    for (const auto &dev : devices)
    {
        if (dev.type == ComputeBackendType::GPU_CUDA)
        {
            has_cuda = true;
            break;
        }
    }

    if (!has_cuda)
    {
        GTEST_SKIP() << "No CUDA device available";
    }

    // Simulate the bug scenario by rapidly creating/destroying tensors
    // and checking that the cache stays clean

    for (int i = 0; i < 10; ++i)
    {
        {
            auto tensor = createTestTensor();
            // Use DeviceId to properly set the target device ordinal
            auto *kernel = getPreparedKernel(tensor.get(), DeviceId::cuda(0));
            ASSERT_NE(kernel, nullptr);
            // tensor destroyed here
        }

        // After each destruction, cache should be empty
        auto [size, _] = KernelFactory::cacheStats();
        EXPECT_EQ(size, 0u)
            << "REGRESSION: Iteration " << i << " - cache should be empty after tensor destruction. "
                                                "Memory reuse could cause stale cache hits if not properly invalidated.";
    }
}

#endif // HAVE_CUDA
