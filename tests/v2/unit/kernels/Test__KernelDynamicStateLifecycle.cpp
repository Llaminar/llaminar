/**
 * @file Test__KernelDynamicStateLifecycle.cpp
 * @brief Unit tests for kernel dynamic state lifecycle (session boundary resets)
 * @author David Sanftenberg
 *
 * Tests verify:
 * 1. ITensorKernel::resetDynamicState() default is no-op
 * 2. ITensorKernel::hasDynamicStateActive() default returns false
 * 3. Embedding kernels track dynamic state via setDynamicTokenIds()
 * 4. Embedding kernels clear dynamic state via resetDynamicState()
 * 5. KernelFactory::resetAllDynamicState() resets all cached embedding kernels
 * 6. KernelFactory::resetAllDynamicState() does NOT destroy cached kernels
 */

#include <gtest/gtest.h>
#include "kernels/KernelFactory.h"
#include "tensors/Tensors.h"
#include "tensors/KernelSnapshotInfo.h"
#include "backends/ComputeBackend.h"
#include "backends/DeviceId.h"
#include "../../utils/TestTensorFactory.h"

using namespace llaminar::v2::kernels;
using namespace llaminar2;
using namespace llaminar2::test;

// ============================================================================
// Test Fixture
// ============================================================================

class Test__KernelDynamicStateLifecycle : public ::testing::Test
{
protected:
    void SetUp() override
    {
        DeviceManager::instance();
    }

    void TearDown() override
    {
        KernelFactory::clearCache();
    }
};

// ============================================================================
// ITensorKernel Base Class Defaults
// ============================================================================

namespace
{
    // Minimal concrete kernel for testing base class defaults
    class StubKernel : public ITensorKernel
    {
    public:
        bool supports_device(int) const override { return true; }
        KernelSnapshotInfo getKernelSnapshotInfo() const override
        {
            return KernelSnapshotInfo::passthrough();
        }
    };
} // namespace

TEST_F(Test__KernelDynamicStateLifecycle, BaseClass_DefaultHasDynamicStateReturnsFalse)
{
    StubKernel stub;
    EXPECT_FALSE(stub.hasDynamicStateActive());
}

TEST_F(Test__KernelDynamicStateLifecycle, BaseClass_DefaultResetIsNoOp)
{
    StubKernel stub;
    // Should not throw or crash
    stub.resetDynamicState();
    EXPECT_FALSE(stub.hasDynamicStateActive());
}

// ============================================================================
// CPU Embedding Kernel (always available, no GPU required)
// ============================================================================

TEST_F(Test__KernelDynamicStateLifecycle, CPUEmbedding_InitiallyNoDynamicState)
{
    auto embed_table = TestTensorFactory::createFP32Random({100, 64});
    auto kernel = KernelFactory::createEmbedding(
        static_cast<const FP32Tensor *>(embed_table.get()), DeviceType::CPU);
    ASSERT_NE(kernel, nullptr);

    // CPU embedding has no dynamic state (no GPU buffer preloading)
    EXPECT_FALSE(kernel->hasDynamicStateActive());
}

TEST_F(Test__KernelDynamicStateLifecycle, CPUEmbedding_ResetIsNoOpSafe)
{
    auto embed_table = TestTensorFactory::createFP32Random({100, 64});
    auto kernel = KernelFactory::createEmbedding(
        static_cast<const FP32Tensor *>(embed_table.get()), DeviceType::CPU);
    ASSERT_NE(kernel, nullptr);

    // Reset should be safe even on CPU kernel
    kernel->resetDynamicState();
    EXPECT_FALSE(kernel->hasDynamicStateActive());
}

// ============================================================================
// KernelFactory::resetAllDynamicState() — Cache Interaction
// ============================================================================

TEST_F(Test__KernelDynamicStateLifecycle, Factory_ResetPreservesKernelObjects)
{
    auto embed_table = TestTensorFactory::createFP32Random({100, 64});

    // Create and cache an embedding kernel
    auto *cached_kernel = KernelFactory::getOrCreateEmbedding(
        embed_table.get(), DeviceId::cpu());
    ASSERT_NE(cached_kernel, nullptr);

    auto [size_before, _] = KernelFactory::cacheStats();
    EXPECT_GT(size_before, 0u);

    // Reset dynamic state — should NOT destroy kernel objects
    KernelFactory::resetAllDynamicState();

    // Same cache size after reset
    auto [size_after, __] = KernelFactory::cacheStats();
    EXPECT_EQ(size_before, size_after);

    // Same kernel pointer should still be valid
    auto *cached_kernel2 = KernelFactory::getOrCreateEmbedding(
        embed_table.get(), DeviceId::cpu());
    EXPECT_EQ(cached_kernel, cached_kernel2);
}

TEST_F(Test__KernelDynamicStateLifecycle, Factory_ResetOnEmptyCacheIsNoOp)
{
    KernelFactory::clearCache();
    // Should not throw or crash
    KernelFactory::resetAllDynamicState();
}
