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
#include "kernels/cpu/native_vnni/CPUNativeVNNIGemmKernel.h"
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

// Register a kernel under a GPU device ID using the GPU pipeline API.
// This simulates what DeviceLoadPipeline/WeightVRAMPool does in production.
static ITensorGemm *registerAndGetGPUKernel(const TensorBase *tensor, DeviceId device)
{
    auto kernel = std::make_unique<llaminar2::cpu::native_vnni::CPUNativeVNNIGemmKernel>(tensor);
    auto *handle = KernelFactory::registerPreparedGemmFromTransfer(tensor, device, std::move(kernel));
    if (!handle) return nullptr;
    return KernelFactory::getOrCreateGemmEngine(handle);
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

    // Verify cache now has entries (prepared_gemm_registry_ uses dual-key insertion,
    // so each tensor may create more than 1 registry entry)
    auto [size_after, bytes] = KernelFactory::cacheStats();
    EXPECT_GT(size_after, 0u);
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
        EXPECT_GT(size_during, 0u);

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

    // Should have entries for both tensors
    auto [size_both, _] = KernelFactory::cacheStats();
    EXPECT_GT(size_both, 0u);

    // Destroy tensor1
    tensor1.reset();

    // Should have fewer entries (tensor2 remains)
    auto [size_one, __] = KernelFactory::cacheStats();
    EXPECT_GT(size_one, 0u) << "tensor2's kernel should remain in cache";
    EXPECT_LT(size_one, size_both) << "Destroying tensor1 should reduce cache size";

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
    EXPECT_GT(size_before, 0u);

    // Create a different tensor (not cached) and clear its entry
    // This should be a no-op - the cache should still have the original entry
    auto tensor2 = createTestTensor();
    KernelFactory::clearCacheFor(tensor2.get());

    // Cache should still have the first tensor's entry
    auto [size_after, __] = KernelFactory::cacheStats();
    EXPECT_EQ(size_after, size_before) << "Cache should be unaffected by clearCacheFor with different tensor";
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
        EXPECT_GT(size_during, 0u);
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
        EXPECT_GT(size_during, 0u);
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
    // NativeVNNI kernels manage their own weight packing internally.
    // tensor->cache_ is NOT used for CPU packed weights anymore.

    {
        auto tensor = createTestTensor();

        // cache_ should remain empty since NativeVNNI packs internally
        EXPECT_FALSE(tensor->cache_.has_value()) << "Fresh tensor should have no cache";

        auto *kernel = getPreparedKernel(tensor.get());
        ASSERT_NE(kernel, nullptr);

        // cache_ stays empty — NativeVNNI packs weights in the kernel itself
        EXPECT_FALSE(tensor->cache_.has_value())
            << "NativeVNNI kernels don't use tensor->cache_ for CPU packed weights";
    }

    auto [size_after, _] = KernelFactory::cacheStats();
    EXPECT_EQ(size_after, 0u) << "Cache should be empty after tensor destruction";
}

TEST_F(Test__KernelFactoryCacheInvalidation, CPUPackedWeights_ClearedByExplicitClearCacheFor)
{
    // NativeVNNI kernels don't populate tensor->cache_, so clearCacheFor is a no-op for CPU.
    auto tensor = createTestTensor();

    getPreparedKernel(tensor.get());
    // cache_ stays empty with NativeVNNI
    EXPECT_FALSE(tensor->cache_.has_value())
        << "NativeVNNI kernels don't use tensor->cache_";

    KernelFactory::clearCacheFor(tensor.get());
    EXPECT_FALSE(tensor->cache_.has_value()) << "clearCacheFor should leave cache_ empty";
}

TEST_F(Test__KernelFactoryCacheInvalidation, CPUPackedWeights_ClearedByClearCache)
{
    // NativeVNNI kernels don't populate tensor->cache_
    auto tensor1 = createTestTensor();
    auto tensor2 = createTestTensor();

    getPreparedKernel(tensor1.get());
    getPreparedKernel(tensor2.get());

    // cache_ stays empty with NativeVNNI
    EXPECT_FALSE(tensor1->cache_.has_value());
    EXPECT_FALSE(tensor2->cache_.has_value());

    KernelFactory::clearCache();

    EXPECT_FALSE(tensor1->cache_.has_value());
    EXPECT_FALSE(tensor2->cache_.has_value());
}

TEST_F(Test__KernelFactoryCacheInvalidation, MultiplePackedWeightsCreation_OnlyPacksOnce)
{
    // KernelFactory still caches the kernel object itself
    auto tensor = createTestTensor();

    auto *kernel1 = getPreparedKernel(tensor.get());
    ASSERT_NE(kernel1, nullptr);

    auto *kernel2 = getPreparedKernel(tensor.get());
    EXPECT_EQ(kernel1, kernel2) << "Should return cached kernel";
}

#ifdef HAVE_CUDA
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
    // 1. Create tensor A, register device-targeted CUDA kernel (cached by tensor ptr)
    // 2. Destroy tensor A (cache entry SHOULD be removed)
    // 3. Create tensor B at same memory address (memory reuse)
    // 4. Query cache for tensor B -> WITHOUT FIX: returns stale kernel from A
    //                                WITH FIX: cache miss, creates new kernel

    {
        auto tensor = createTestTensor();

        // Register a GPU kernel via the production GPU pipeline API
        auto *kernel = registerAndGetGPUKernel(tensor.get(), DeviceId::cuda(0));
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

    auto tensor1 = createTestTensor();
    auto tensor2 = createTestTensor();

    // Register GPU kernels for both tensors via the GPU pipeline API
    auto *kernel1 = registerAndGetGPUKernel(tensor1.get(), DeviceId::cuda(0));
    auto *kernel2 = registerAndGetGPUKernel(tensor2.get(), DeviceId::cuda(0));

    ASSERT_NE(kernel1, nullptr);
    ASSERT_NE(kernel2, nullptr);

    auto [size_both, _] = KernelFactory::cacheStats();
    EXPECT_GE(size_both, 2u) << "Should have at least 2 cache entries";

    // Destroy tensor1
    tensor1.reset();

    // tensor2's kernel should still be in cache
    auto [size_one, __] = KernelFactory::cacheStats();
    EXPECT_GE(size_one, 1u) << "tensor2's kernel should still be cached";

    // Verify we can still get tensor2's kernel (cache hit via getPreparedKernel)
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

    {
        auto tensor = createTestTensor();

        // Create CPU kernel via normal path, CUDA kernel via GPU pipeline API
        auto *cpu_kernel = getPreparedKernel(tensor.get(), DeviceId::cpu());
        auto *cuda_kernel = registerAndGetGPUKernel(tensor.get(), DeviceId::cuda(0));

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

    auto tensor = createTestTensor();

    // Register a GPU kernel via the GPU pipeline API
    auto *kernel = registerAndGetGPUKernel(tensor.get(), DeviceId::cuda(0));
    ASSERT_NE(kernel, nullptr);

    auto [size_before, _] = KernelFactory::cacheStats();
    EXPECT_GE(size_before, 1u);

    // Explicitly clear cache for this tensor
    KernelFactory::clearCacheFor(tensor.get());

    // Entry should be removed
    auto [size_after, __] = KernelFactory::cacheStats();
    EXPECT_EQ(size_after, 0u)
        << "clearCacheFor should remove device_targeted_cache_ entries";

    // Re-register a new kernel after cache clear
    auto *kernel_new = registerAndGetGPUKernel(tensor.get(), DeviceId::cuda(0));
    ASSERT_NE(kernel_new, nullptr);
    // The key point is the cache was cleared and re-population works
}

TEST_F(Test__KernelFactoryCacheInvalidation, DeviceTargetedCache_MemoryReuseSimulation)
{
    // This test simulates the exact scenario that caused the original bug:
    // Memory reuse after tensor destruction leading to stale cache hits.
    //
    // We can't force memory reuse, but we can verify the cache is clean
    // after destruction, which prevents the bug.

    // Simulate the bug scenario by rapidly creating/destroying tensors
    // and checking that the cache stays clean

    for (int i = 0; i < 10; ++i)
    {
        {
            auto tensor = createTestTensor();
            // Register GPU kernel via the production GPU pipeline API
            auto *kernel = registerAndGetGPUKernel(tensor.get(), DeviceId::cuda(0));
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
