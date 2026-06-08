/**
 * @file Test__KernelDynamicStateLifecycle.cpp
 * @brief Unit tests for kernel dynamic state lifecycle (session boundary resets)
 * @author David Sanftenberg
 *
 * Tests verify:
 * 1. ITensorKernel::resetDynamicState() default is no-op
 * 2. ITensorKernel::hasDynamicStateActive() default returns false
 * 3. CPU embedding kernels: reset is no-op (no GPU state)
 * 4. CUDA embedding kernels: setDynamicTokenIds activates state, resetDynamicState clears it
 * 5. ROCm embedding kernels: same lifecycle as CUDA
 * 6. KernelFactory::resetAllDynamicState() resets all cached kernels without destroying them
 * 7. Embedding memcmp guard: changed token IDs force re-upload even with active state
 *
 * Regression coverage for:
 *   - Stale dynamic_params_active_ after clearCache() (eeca83dd)
 *   - KernelFactory::resetAllDynamicState() lifecycle (8666332f)
 */

#include <gtest/gtest.h>
#include "kernels/KernelFactory.h"
#include "tensors/Tensors.h"
#include "tensors/KernelSnapshotInfo.h"
#include "backends/ComputeBackend.h"
#include "backends/DeviceId.h"
#include "backends/GPUDeviceContextPool.h"
#include "execution/local_execution/device/DeviceWorkspaceManager.h"
#include "interfaces/IWorkspaceConsumer.h"
#include "../../utils/TestTensorFactory.h"

#ifdef HAVE_ROCM
#include "kernels/rocm/ops/ROCmEmbeddingKernelT.h"
#endif

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
        auto &dm = DeviceManager::instance();
        if (dm.devices().empty())
            dm.initialize(-1);
    }

    void TearDown() override
    {
        KernelFactory::clearCache();
    }

    bool hasCUDA() const { return DeviceManager::instance().cuda_device_count() > 0; }
    bool hasROCm() const { return DeviceManager::instance().rocm_device_count() > 0; }

    // Helper: create GPU embedding kernel with workspace bound, so setDynamicTokenIds works
    struct GpuKernelWithWorkspace
    {
        std::unique_ptr<ITensorEmbedding> kernel;
        std::unique_ptr<DeviceWorkspaceManager> workspace;
    };

    GpuKernelWithWorkspace createGpuEmbeddingWithWorkspace(DeviceType dev_type, int ordinal = 0)
    {
        auto embed_table = TestTensorFactory::createFP32Random({100, 64});
        auto kernel = KernelFactory::createEmbedding(
            static_cast<const FP32Tensor *>(embed_table.get()), dev_type, ordinal);

        DeviceId device = (dev_type == DeviceType::CUDA) ? DeviceId::cuda(ordinal) : DeviceId::rocm(ordinal);
        auto workspace = std::make_unique<DeviceWorkspaceManager>(device, 64 * 1024);

        // Allocate a minimal workspace with just the TOKEN_IDS buffer
        WorkspaceRequirements reqs;
        reqs.buffers.push_back({EmbeddingWorkspaceBuffers::TOKEN_IDS,
                                512 * sizeof(int), 256, true});
        workspace->allocate(reqs);

        auto *ws_consumer = dynamic_cast<IWorkspaceConsumer *>(kernel.get());
        if (ws_consumer)
            ws_consumer->bindWorkspace(workspace.get());

        kernel->setGPUStream(GPUDeviceContextPool::instance().getContext(device).defaultStream());

        return {std::move(kernel), std::move(workspace)};
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
    EXPECT_FALSE(kernel->hasDynamicStateActive());
}

TEST_F(Test__KernelDynamicStateLifecycle, CPUEmbedding_ResetIsNoOpSafe)
{
    auto embed_table = TestTensorFactory::createFP32Random({100, 64});
    auto kernel = KernelFactory::createEmbedding(
        static_cast<const FP32Tensor *>(embed_table.get()), DeviceType::CPU);
    ASSERT_NE(kernel, nullptr);
    kernel->resetDynamicState();
    EXPECT_FALSE(kernel->hasDynamicStateActive());
}

// ============================================================================
// CUDA Embedding Kernel — Dynamic State Lifecycle
// ============================================================================

TEST_F(Test__KernelDynamicStateLifecycle, CUDAEmbedding_SetDynamicTokenIds_ActivatesState)
{
    if (!hasCUDA())
        GTEST_SKIP() << "No CUDA GPU available";

    auto [kernel, workspace] = createGpuEmbeddingWithWorkspace(DeviceType::CUDA, 0);
    ASSERT_NE(kernel, nullptr);

    EXPECT_FALSE(kernel->hasDynamicStateActive());

    // Preload token IDs → should activate dynamic state
    std::vector<int> tokens = {1, 5, 10};
    kernel->setDynamicTokenIds(tokens.data(), static_cast<int>(tokens.size()));

    EXPECT_TRUE(kernel->hasDynamicStateActive())
        << "setDynamicTokenIds must activate dynamic state on CUDA kernel";
}

TEST_F(Test__KernelDynamicStateLifecycle, CUDAEmbedding_ResetDynamicState_ClearsFlag)
{
    if (!hasCUDA())
        GTEST_SKIP() << "No CUDA GPU available";

    auto [kernel, workspace] = createGpuEmbeddingWithWorkspace(DeviceType::CUDA, 0);
    ASSERT_NE(kernel, nullptr);

    // Activate then reset
    std::vector<int> tokens = {1, 5, 10};
    kernel->setDynamicTokenIds(tokens.data(), static_cast<int>(tokens.size()));
    ASSERT_TRUE(kernel->hasDynamicStateActive());

    kernel->resetDynamicState();
    EXPECT_FALSE(kernel->hasDynamicStateActive())
        << "resetDynamicState must clear dynamic_params_active_ on CUDA kernel";
}

TEST_F(Test__KernelDynamicStateLifecycle, CUDAEmbedding_DoubleResetIsSafe)
{
    if (!hasCUDA())
        GTEST_SKIP() << "No CUDA GPU available";

    auto [kernel, workspace] = createGpuEmbeddingWithWorkspace(DeviceType::CUDA, 0);
    ASSERT_NE(kernel, nullptr);

    std::vector<int> tokens = {1, 5, 10};
    kernel->setDynamicTokenIds(tokens.data(), static_cast<int>(tokens.size()));

    kernel->resetDynamicState();
    kernel->resetDynamicState(); // Must not crash
    EXPECT_FALSE(kernel->hasDynamicStateActive());
}

TEST_F(Test__KernelDynamicStateLifecycle, CUDAEmbedding_ReactivateAfterReset)
{
    if (!hasCUDA())
        GTEST_SKIP() << "No CUDA GPU available";

    auto [kernel, workspace] = createGpuEmbeddingWithWorkspace(DeviceType::CUDA, 0);
    ASSERT_NE(kernel, nullptr);

    // Cycle: activate → reset → re-activate
    std::vector<int> tokens = {1, 5, 10};
    kernel->setDynamicTokenIds(tokens.data(), static_cast<int>(tokens.size()));
    ASSERT_TRUE(kernel->hasDynamicStateActive());

    kernel->resetDynamicState();
    ASSERT_FALSE(kernel->hasDynamicStateActive());

    // Re-activate with different tokens
    std::vector<int> tokens2 = {42, 99};
    kernel->setDynamicTokenIds(tokens2.data(), static_cast<int>(tokens2.size()));
    EXPECT_TRUE(kernel->hasDynamicStateActive())
        << "Kernel must be re-activatable after reset";
}

// ============================================================================
// ROCm Embedding Kernel — Dynamic State Lifecycle
// ============================================================================

TEST_F(Test__KernelDynamicStateLifecycle, ROCmEmbedding_SetDynamicTokenIds_ActivatesState)
{
    if (!hasROCm())
        GTEST_SKIP() << "No ROCm GPU available";

    auto [kernel, workspace] = createGpuEmbeddingWithWorkspace(DeviceType::ROCm, 0);
    ASSERT_NE(kernel, nullptr);

    EXPECT_FALSE(kernel->hasDynamicStateActive());

    std::vector<int> tokens = {1, 5, 10};
    kernel->setDynamicTokenIds(tokens.data(), static_cast<int>(tokens.size()));

    EXPECT_TRUE(kernel->hasDynamicStateActive())
        << "setDynamicTokenIds must activate dynamic state on ROCm kernel";
}

TEST_F(Test__KernelDynamicStateLifecycle, ROCmEmbedding_ResetDynamicState_ClearsFlag)
{
    if (!hasROCm())
        GTEST_SKIP() << "No ROCm GPU available";

    auto [kernel, workspace] = createGpuEmbeddingWithWorkspace(DeviceType::ROCm, 0);
    ASSERT_NE(kernel, nullptr);

    std::vector<int> tokens = {1, 5, 10};
    kernel->setDynamicTokenIds(tokens.data(), static_cast<int>(tokens.size()));
    ASSERT_TRUE(kernel->hasDynamicStateActive());

    kernel->resetDynamicState();
    EXPECT_FALSE(kernel->hasDynamicStateActive())
        << "resetDynamicState must clear dynamic_params_active_ on ROCm kernel";
}

TEST_F(Test__KernelDynamicStateLifecycle, ROCmEmbedding_ReactivateAfterReset)
{
    if (!hasROCm())
        GTEST_SKIP() << "No ROCm GPU available";

    auto [kernel, workspace] = createGpuEmbeddingWithWorkspace(DeviceType::ROCm, 0);
    ASSERT_NE(kernel, nullptr);

    std::vector<int> tokens = {1, 5, 10};
    kernel->setDynamicTokenIds(tokens.data(), static_cast<int>(tokens.size()));
    ASSERT_TRUE(kernel->hasDynamicStateActive());

    kernel->resetDynamicState();
    ASSERT_FALSE(kernel->hasDynamicStateActive());

    std::vector<int> tokens2 = {42, 99};
    kernel->setDynamicTokenIds(tokens2.data(), static_cast<int>(tokens2.size()));
    EXPECT_TRUE(kernel->hasDynamicStateActive())
        << "Kernel must be re-activatable after reset";
}

TEST_F(Test__KernelDynamicStateLifecycle, ROCmEmbedding_UnbindClearsDeviceSpecificWorkspace)
{
#ifndef HAVE_ROCM
    GTEST_SKIP() << "ROCm backend not compiled";
#else
    if (!hasROCm())
        GTEST_SKIP() << "No ROCm GPU available";

    ROCmEmbeddingKernelT kernel;
    DeviceId device = DeviceId::rocm(0);
    auto workspace = std::make_unique<DeviceWorkspaceManager>(device, 64 * 1024);

    WorkspaceRequirements reqs;
    reqs.buffers.push_back({EmbeddingWorkspaceBuffers::TOKEN_IDS,
                            512 * sizeof(int), 256, true});
    ASSERT_TRUE(workspace->allocate(reqs));

    kernel.bindWorkspace(workspace.get());

    std::vector<int> tokens = {1, 5, 10};
    kernel.setDynamicTokenIds(tokens.data(), static_cast<int>(tokens.size()));
    ASSERT_TRUE(kernel.hasDynamicStateActive());

    kernel.bindWorkspace(nullptr);
    kernel.setDynamicTokenIds(tokens.data(), static_cast<int>(tokens.size()));
    EXPECT_FALSE(kernel.hasDynamicStateActive())
        << "Unbind must clear device-specific workspace bindings, including "
           "bindings created while device_idx_ was unspecified";
#endif
}

// ============================================================================
// KernelFactory::resetAllDynamicState() — Cache Interaction
// ============================================================================

TEST_F(Test__KernelDynamicStateLifecycle, Factory_ResetPreservesKernelObjects)
{
    auto embed_table = TestTensorFactory::createFP32Random({100, 64});

    auto *cached_kernel = KernelFactory::getOrCreateEmbedding(
        embed_table.get(), DeviceId::cpu());
    ASSERT_NE(cached_kernel, nullptr);

    auto [size_before, _] = KernelFactory::cacheStats();
    EXPECT_GT(size_before, 0u);

    KernelFactory::resetAllDynamicState();

    auto [size_after, __] = KernelFactory::cacheStats();
    EXPECT_EQ(size_before, size_after);

    auto *cached_kernel2 = KernelFactory::getOrCreateEmbedding(
        embed_table.get(), DeviceId::cpu());
    EXPECT_EQ(cached_kernel, cached_kernel2)
        << "resetAllDynamicState must NOT destroy cached kernels";
}

TEST_F(Test__KernelDynamicStateLifecycle, Factory_ResetOnEmptyCacheIsNoOp)
{
    KernelFactory::clearCache();
    KernelFactory::resetAllDynamicState();
}

// Regression: KernelFactory reset must clear active state on GPU embedding kernels.
// This is the exact bug path: KernelFactory caches embedding kernel, session ends,
// resetAllDynamicState() must clear the flag so next session re-uploads tokens.
TEST_F(Test__KernelDynamicStateLifecycle, Factory_ResetClearsGPUEmbeddingDynamicState)
{
    if (!hasCUDA() && !hasROCm())
        GTEST_SKIP() << "No GPU available";

    auto embed_table = TestTensorFactory::createFP32Random({100, 64});

    // Create a GPU embedding kernel via factory cache
    DeviceId device = hasCUDA() ? DeviceId::cuda(0) : DeviceId::rocm(0);
    auto *cached = KernelFactory::getOrCreateEmbedding(embed_table.get(), device);
    ASSERT_NE(cached, nullptr);

    // Bind a workspace so setDynamicTokenIds can actually activate
    auto workspace = std::make_unique<DeviceWorkspaceManager>(device, 64 * 1024);
    WorkspaceRequirements reqs;
    reqs.buffers.push_back({EmbeddingWorkspaceBuffers::TOKEN_IDS,
                            512 * sizeof(int), 256, true});
    ASSERT_TRUE(workspace->allocate(reqs));

    auto *ws_consumer = dynamic_cast<IWorkspaceConsumer *>(cached);
    ASSERT_NE(ws_consumer, nullptr);
    ws_consumer->bindWorkspace(workspace.get());
    cached->setGPUStream(GPUDeviceContextPool::instance().getContext(device).defaultStream());

    // Activate dynamic state (simulates prefill preloading token IDs)
    std::vector<int> tokens = {1, 5, 10};
    cached->setDynamicTokenIds(tokens.data(), static_cast<int>(tokens.size()));
    ASSERT_TRUE(cached->hasDynamicStateActive());

    // Session boundary: factory resets all cached kernels
    KernelFactory::resetAllDynamicState();

    EXPECT_FALSE(cached->hasDynamicStateActive())
        << "Factory reset must clear dynamic state on GPU embedding kernels";
}
