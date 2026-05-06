/**
 * @file Test__EmbeddingKernelPreparedPath.cpp
 * @brief Unit tests for embedding kernel workspace and prepared path selection
 *
 * Tests verify:
 * 1. TOKEN_IDS workspace buffer is always requested
 * 2. EMBED_TABLE workspace buffer is requested when no prepared embedding handle is injected
 * 3. EMBED_TABLE workspace buffer is skipped when a prepared embedding handle is injected
 * 4. GPU embedding kernel can be created (GTEST_SKIP if no GPU)
 *
 * These tests inject explicit prepared embedding handles to verify workspace requirements.
 * GPU tests are skipped if no CUDA/ROCm device is available.
 */

#include <gtest/gtest.h>
#include <algorithm>
#include <memory>

#include "kernels/KernelFactory.h"
#include "kernels/common/PreparedEmbeddingWeights.h"
#include "tensors/Tensors.h"
#include "interfaces/IWorkspaceConsumer.h"
#include "execution/local_execution/device/WorkspaceDescriptor.h"
#include "backends/ComputeBackend.h"
#include "backends/BackendManager.h"
#include "../../utils/TestTensorFactory.h"

using namespace llaminar::v2::kernels;
using namespace llaminar2;
using namespace llaminar2::test;

// ============================================================================
// Test Fixture
// ============================================================================

class Test__EmbeddingKernelPreparedPath : public ::testing::Test
{
protected:
    void SetUp() override
    {
        auto &dm = DeviceManager::instance();
        if (dm.devices().empty())
            dm.initialize(-1);

        if (!getCPUBackend())
            initCPUBackend(0);

        KernelFactory::clearCache();
    }

    void TearDown() override
    {
        KernelFactory::clearCache();
    }

    bool hasCUDA() const { return DeviceManager::instance().cuda_device_count() > 0; }
    bool hasROCm() const { return DeviceManager::instance().rocm_device_count() > 0; }
    bool hasGPU() const { return hasCUDA() || hasROCm(); }

    DeviceType gpuDeviceType() const
    {
        if (hasROCm())
            return DeviceType::ROCm;
        return DeviceType::CUDA;
    }

    DeviceId gpuDeviceId() const
    {
        if (hasROCm())
            return DeviceId::rocm(0);
        return DeviceId::cuda(0);
    }

    /// Helper: check if workspace requirements contain a buffer with the given name
    static bool hasBuffer(const WorkspaceRequirements &reqs, const char *name)
    {
        return std::any_of(reqs.buffers.begin(), reqs.buffers.end(),
                           [name](const WorkspaceDescriptor &d)
                           { return d.name == name; });
    }
};

// ============================================================================
// GPU Workspace Requirements Tests
// ============================================================================

TEST_F(Test__EmbeddingKernelPreparedPath, TokenIDsAlwaysRequested)
{
    if (!hasGPU())
        GTEST_SKIP() << "No GPU available";

    auto embed_table = TestTensorFactory::createFP32Random({100, 64});
    auto kernel = KernelFactory::createEmbedding(
        static_cast<const FP32Tensor *>(embed_table.get()), gpuDeviceType(), 0);
    ASSERT_NE(kernel, nullptr);

    auto *ws_consumer = dynamic_cast<IWorkspaceConsumer *>(kernel.get());
    ASSERT_NE(ws_consumer, nullptr);

    auto reqs = ws_consumer->getWorkspaceRequirements(512, 100, 64);

    EXPECT_TRUE(hasBuffer(reqs, EmbeddingWorkspaceBuffers::TOKEN_IDS));
}

TEST_F(Test__EmbeddingKernelPreparedPath, EmbedTableRequestedWhenNoPrepared)
{
    if (!hasGPU())
        GTEST_SKIP() << "No GPU available";

    auto embed_table = TestTensorFactory::createFP32Random({100, 64});
    auto kernel = KernelFactory::createEmbedding(
        static_cast<const FP32Tensor *>(embed_table.get()), gpuDeviceType(), 0);
    ASSERT_NE(kernel, nullptr);

    auto *ws_consumer = dynamic_cast<IWorkspaceConsumer *>(kernel.get());
    ASSERT_NE(ws_consumer, nullptr);

    auto reqs = ws_consumer->getWorkspaceRequirements(512, 100, 64);

    EXPECT_TRUE(hasBuffer(reqs, EmbeddingWorkspaceBuffers::TOKEN_IDS));
    EXPECT_TRUE(hasBuffer(reqs, EmbeddingWorkspaceBuffers::EMBED_TABLE));
}

TEST_F(Test__EmbeddingKernelPreparedPath, EmbedTableSkippedWhenPreparedExists)
{
    if (!hasGPU())
        GTEST_SKIP() << "No GPU available";

    auto embed_table = TestTensorFactory::createFP32Random({100, 64});
    auto kernel = KernelFactory::createEmbedding(
        static_cast<const FP32Tensor *>(embed_table.get()), gpuDeviceType(), 0);
    ASSERT_NE(kernel, nullptr);

    PreparedEmbeddingHandle handle;
    handle.tensor = embed_table.get();
    handle.device_id = gpuDeviceId();
    handle.weights = std::make_shared<PreparedEmbeddingWeights>();
    kernel->setPreparedEmbeddingHandle(&handle);

    auto *ws_consumer = dynamic_cast<IWorkspaceConsumer *>(kernel.get());
    ASSERT_NE(ws_consumer, nullptr);

    auto reqs = ws_consumer->getWorkspaceRequirements(512, 100, 64);

    EXPECT_TRUE(hasBuffer(reqs, EmbeddingWorkspaceBuffers::TOKEN_IDS));
    EXPECT_FALSE(hasBuffer(reqs, EmbeddingWorkspaceBuffers::EMBED_TABLE))
        << "EMBED_TABLE should be skipped when PreparedEmbeddingWeights exist";
}

TEST_F(Test__EmbeddingKernelPreparedPath, EmbedTableRequestedWithoutInjectedPreparedHandle)
{
    if (!hasGPU())
        GTEST_SKIP() << "No GPU available";

    auto embed_table = TestTensorFactory::createFP32Random({100, 64});
    auto kernel = KernelFactory::createEmbedding(
        static_cast<const FP32Tensor *>(embed_table.get()), gpuDeviceType(), 0);
    ASSERT_NE(kernel, nullptr);

    auto *ws_consumer = dynamic_cast<IWorkspaceConsumer *>(kernel.get());
    ASSERT_NE(ws_consumer, nullptr);

    auto reqs = ws_consumer->getWorkspaceRequirements(512, 100, 64);

    EXPECT_TRUE(hasBuffer(reqs, EmbeddingWorkspaceBuffers::EMBED_TABLE))
        << "EMBED_TABLE should be requested when no prepared handle is injected";
}

// ============================================================================
// CPU Embedding Tests (no GPU required)
// ============================================================================

TEST_F(Test__EmbeddingKernelPreparedPath, CPUEmbeddingCreatable)
{
    auto embed_table = TestTensorFactory::createFP32Random({100, 64});
    auto kernel = KernelFactory::createEmbedding(
        static_cast<const FP32Tensor *>(embed_table.get()), DeviceType::CPU);

    ASSERT_NE(kernel, nullptr);
}

TEST_F(Test__EmbeddingKernelPreparedPath, CPUEmbeddingHasNoWorkspaceRequirements)
{
    auto embed_table = TestTensorFactory::createFP32Random({100, 64});
    auto kernel = KernelFactory::createEmbedding(
        static_cast<const FP32Tensor *>(embed_table.get()), DeviceType::CPU);
    ASSERT_NE(kernel, nullptr);

    // CPU embedding doesn't implement IWorkspaceConsumer (or returns empty reqs)
    auto *ws_consumer = dynamic_cast<IWorkspaceConsumer *>(kernel.get());
    if (ws_consumer)
    {
        auto reqs = ws_consumer->getWorkspaceRequirements(512, 100, 64);
        EXPECT_TRUE(reqs.buffers.empty())
            << "CPU embedding should have no workspace requirements";
    }
    // If ws_consumer is null, that's also acceptable (CPU doesn't need workspace)
}
