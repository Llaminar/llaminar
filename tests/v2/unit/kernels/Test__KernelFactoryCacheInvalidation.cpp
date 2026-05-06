/**
 * @file Test__KernelFactoryCacheInvalidation.cpp
 * @brief Unit tests for KernelFactory cache invalidation APIs
 * @author David Sanftenberg
 *
 * Phase 10 (Weight Lifetime Redesign): TensorBase destructors no longer
 * auto-clear KernelFactory caches. Instead, explicit cleanup is performed via:
 *   - KernelFactory::clearPreparedStateForTensor() — per-tensor cleanup
 *   - KernelFactory::clearCacheFor() — legacy per-tensor cleanup
 *   - PreparedWeightStore::releaseAllPreparedState() — bulk model teardown
 *
 * These tests verify that explicit cleanup APIs work correctly, and that
 * the cache retains entries across tensor destruction (the new contract).
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
    auto kernel = std::make_unique<llaminar2::cpu::native_vnni::CPUNativeVNNIGemmKernel>(tensor);
    auto *handle = KernelFactory::registerPreparedGemmFromTransfer(tensor, device, std::move(kernel));
    if (!handle)
        return nullptr;
    return KernelFactory::getOrCreateGemmEngine(handle);
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

TEST_F(Test__KernelFactoryCacheInvalidation, CacheRetainedAfterTensorDestruction)
{
    // Phase 10: TensorBase destructor no longer clears KernelFactory cache.
    // Cache entries persist until explicit cleanup via clearPreparedStateForTensor().

    const TensorBase *captured_ptr = nullptr;

    // Create tensor in a scope
    {
        auto tensor = createTestTensor();
        captured_ptr = tensor.get();

        // Add to cache
        auto *kernel = getPreparedKernel(tensor.get());
        ASSERT_NE(kernel, nullptr);

        // Verify it's in the cache
        auto [size_during, _] = KernelFactory::cacheStats();
        EXPECT_GT(size_during, 0u);

        // tensor goes out of scope here -> destructor runs but does NOT clear cache
    }

    // Phase 10: Cache entry persists after tensor destruction
    auto [size_after, bytes_after] = KernelFactory::cacheStats();
    EXPECT_GT(size_after, 0u) << "Phase 10: cache should persist after tensor destruction";

    // Explicit cleanup removes the entry
    KernelFactory::clearPreparedStateForTensor(captured_ptr);
    auto [size_cleaned, _] = KernelFactory::cacheStats();
    EXPECT_EQ(size_cleaned, 0u) << "Explicit clearPreparedStateForTensor should remove entry";
}

TEST_F(Test__KernelFactoryCacheInvalidation, MultipleTensors_ExplicitInvalidation)
{
    // Phase 10: Explicit cleanup removes specific tensor entries

    auto tensor1 = createTestTensor();
    auto tensor2 = createTestTensor();

    // Add both to cache
    getPreparedKernel(tensor1.get());
    getPreparedKernel(tensor2.get());

    // Should have entries for both tensors
    auto [size_both, _] = KernelFactory::cacheStats();
    EXPECT_GT(size_both, 0u);

    // Explicit cleanup for tensor1
    KernelFactory::clearPreparedStateForTensor(tensor1.get());

    // Should have fewer entries (tensor2 remains)
    auto [size_one, __] = KernelFactory::cacheStats();
    EXPECT_GT(size_one, 0u) << "tensor2's kernel should remain in cache";
    EXPECT_LT(size_one, size_both) << "Clearing tensor1 should reduce cache size";

    // Explicit cleanup for tensor2
    KernelFactory::clearPreparedStateForTensor(tensor2.get());

    // Cache should now be empty
    auto [size_none, ___] = KernelFactory::cacheStats();
    EXPECT_EQ(size_none, 0u) << "Cache should be empty after explicit cleanup of all tensors";
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

TEST_F(Test__KernelFactoryCacheInvalidation, MemoryReuse_ExplicitCleanupPreventsStaleEntries)
{
    // Phase 10: Explicit cleanup before tensor destruction prevents stale entries.
    // This simulates what PreparedWeightStore::releaseAllPreparedState() does.

    const TensorBase *captured_ptr = nullptr;

    // Create and cache tensor
    {
        auto tensor = createTestTensor();
        captured_ptr = tensor.get();

        getPreparedKernel(tensor.get());

        auto [size_during, _] = KernelFactory::cacheStats();
        EXPECT_GT(size_during, 0u);

        // Explicit cleanup BEFORE destruction (the Phase 10 pattern)
        KernelFactory::clearPreparedStateForTensor(tensor.get());
    }

    // After explicit cleanup + destruction, cache is clean
    auto [size_after, _] = KernelFactory::cacheStats();
    EXPECT_EQ(size_after, 0u) << "Cache should be clean after explicit cleanup";
}

// ============================================================================
// FP32 Tensor (non-quantized) Test
// ============================================================================

TEST_F(Test__KernelFactoryCacheInvalidation, FP32Tensor_ExplicitInvalidation)
{
    // FP32 tensors: explicit cleanup works
    const TensorBase *captured_ptr = nullptr;
    {
        const size_t rows = 32;
        const size_t cols = 32;
        auto tensor = std::make_unique<FP32Tensor>(std::vector<size_t>{rows, cols});
        captured_ptr = tensor.get();

        auto *kernel = getPreparedKernel(tensor.get());
        ASSERT_NE(kernel, nullptr);

        auto [size_during, _] = KernelFactory::cacheStats();
        EXPECT_GT(size_during, 0u);

        // Explicit cleanup before destruction
        KernelFactory::clearPreparedStateForTensor(tensor.get());
    }

    // After explicit cleanup, cache should be empty
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

TEST_F(Test__KernelFactoryCacheInvalidation, RapidCreateDestroy_ExplicitCleanup)
{
    // Phase 10: explicit cleanup keeps cache bounded during rapid cycles
    for (int i = 0; i < 100; ++i)
    {
        auto tensor = createTestTensor();
        getPreparedKernel(tensor.get());
        KernelFactory::clearPreparedStateForTensor(tensor.get());
    }

    // After all explicit cleanups, cache should be empty
    auto [size, _] = KernelFactory::cacheStats();
    EXPECT_EQ(size, 0u) << "Cache should be clean after rapid create/cleanup cycles";
}

// ============================================================================
// Packed Weights Cache Cleanup Tests
// ============================================================================

TEST_F(Test__KernelFactoryCacheInvalidation, CPUPackedWeights_CleanedUpByExplicitCall)
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

        // Phase 10: explicit cleanup
        KernelFactory::clearPreparedStateForTensor(tensor.get());
    }

    auto [size_after, _] = KernelFactory::cacheStats();
    EXPECT_EQ(size_after, 0u) << "Cache should be empty after explicit cleanup";
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
// Device-Targeted Cache: Explicit Cleanup Tests (Phase 10)
// ============================================================================
// Phase 10: TensorBase destructor no longer auto-clears caches.
// These tests verify that explicit clearPreparedStateForTensor() works for
// device-targeted (GPU) cache entries.

TEST_F(Test__KernelFactoryCacheInvalidation, DeviceTargetedCache_CUDA_ExplicitCleanup)
{
    // Phase 10: explicit cleanup removes device-targeted entries

    auto tensor = createTestTensor();

    // Register a GPU kernel via the production GPU pipeline API
    auto *kernel = registerAndGetGPUKernel(tensor.get(), DeviceId::cuda(0));
    ASSERT_NE(kernel, nullptr);

    // Verify cache has an entry
    auto [size_during, _] = KernelFactory::cacheStats();
    EXPECT_GE(size_during, 1u) << "Cache should have at least one entry";

    // Explicit cleanup
    KernelFactory::clearPreparedStateForTensor(tensor.get());

    auto [size_after, __] = KernelFactory::cacheStats();
    EXPECT_EQ(size_after, 0u)
        << "clearPreparedStateForTensor should remove device-targeted cache entries";
}

TEST_F(Test__KernelFactoryCacheInvalidation, DeviceTargetedCache_CPU_ExplicitCleanup)
{
    // Same test but for CPU device-targeted kernels
    auto tensor = createTestTensor();

    // Create CPU device-targeted kernel
    auto *kernel = getPreparedKernel(tensor.get(), DeviceId::cpu());
    ASSERT_NE(kernel, nullptr);

    auto [size_during, _] = KernelFactory::cacheStats();
    EXPECT_GE(size_during, 1u);

    KernelFactory::clearPreparedStateForTensor(tensor.get());

    auto [size_after, __] = KernelFactory::cacheStats();
    EXPECT_EQ(size_after, 0u)
        << "clearPreparedStateForTensor should remove CPU device-targeted entries";
}

TEST_F(Test__KernelFactoryCacheInvalidation, DeviceTargetedCache_MultipleDevices_IndependentCleanup)
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

    // Explicit cleanup for tensor1 only
    KernelFactory::clearPreparedStateForTensor(tensor1.get());

    // tensor2's kernel should still be in cache
    auto [size_one, __] = KernelFactory::cacheStats();
    EXPECT_GE(size_one, 1u) << "tensor2's kernel should still be cached";

    // Verify we can still get tensor2's kernel (cache hit via getPreparedKernel)
    auto *kernel2_again = getPreparedKernel(tensor2.get(), DeviceId::cuda(0));
    EXPECT_EQ(kernel2, kernel2_again) << "Should return same cached kernel for tensor2";

    // Explicit cleanup for tensor2
    KernelFactory::clearPreparedStateForTensor(tensor2.get());

    // Now cache should be empty
    auto [size_none, ___] = KernelFactory::cacheStats();
    EXPECT_EQ(size_none, 0u) << "Cache should be empty after explicit cleanup of all tensors";
}

TEST_F(Test__KernelFactoryCacheInvalidation, DeviceTargetedCache_SameTensorBothDevices_ExplicitCleanup)
{
    // Test tensor with kernels cached for BOTH CPU and CUDA device types

    auto tensor = createTestTensor();

    // Create CPU kernel via normal path, CUDA kernel via GPU pipeline API
    auto *cpu_kernel = getPreparedKernel(tensor.get(), DeviceId::cpu());
    auto *cuda_kernel = registerAndGetGPUKernel(tensor.get(), DeviceId::cuda(0));

    ASSERT_NE(cpu_kernel, nullptr);
    ASSERT_NE(cuda_kernel, nullptr);
    EXPECT_NE(cpu_kernel, cuda_kernel) << "CPU and CUDA kernels should be different";

    auto [size_during, _] = KernelFactory::cacheStats();
    EXPECT_GE(size_during, 2u) << "Should have entries for both device types";

    // Single explicit cleanup should remove both entries
    KernelFactory::clearPreparedStateForTensor(tensor.get());

    auto [size_after, __] = KernelFactory::cacheStats();
    EXPECT_EQ(size_after, 0u)
        << "clearPreparedStateForTensor should remove both CPU and CUDA entries";
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

TEST_F(Test__KernelFactoryCacheInvalidation, DeviceTargetedCache_RapidExplicitCleanup)
{
    // Phase 10: explicit cleanup keeps cache bounded during rapid cycles
    for (int i = 0; i < 10; ++i)
    {
        auto tensor = createTestTensor();
        auto *kernel = registerAndGetGPUKernel(tensor.get(), DeviceId::cuda(0));
        ASSERT_NE(kernel, nullptr);

        KernelFactory::clearPreparedStateForTensor(tensor.get());

        auto [size, _] = KernelFactory::cacheStats();
        EXPECT_EQ(size, 0u)
            << "Iteration " << i << " - cache should be empty after explicit cleanup";
    }
}

#endif // HAVE_CUDA
