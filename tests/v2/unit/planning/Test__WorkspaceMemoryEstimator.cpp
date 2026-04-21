#include <gtest/gtest.h>
#include "planning/WorkspaceMemoryEstimator.h"
#include "backends/DeviceId.h"

using namespace llaminar2;

TEST(Test__WorkspaceMemoryEstimator, GPU_ReturnsNonZero)
{
    size_t bytes = WorkspaceMemoryEstimator::estimate(
        1, 4096, 896, 4864, 151936, DeviceId::cuda(0));

    EXPECT_GT(bytes, 0u);
    // Should be at least the 768 MB floor
    EXPECT_GE(bytes, 768ULL * 1024 * 1024);
}

TEST(Test__WorkspaceMemoryEstimator, CPU_ReturnsZero)
{
    size_t bytes = WorkspaceMemoryEstimator::estimate(
        1, 4096, 896, 4864, 151936, DeviceId::cpu());

    EXPECT_EQ(bytes, 0u);
}

TEST(Test__WorkspaceMemoryEstimator, GPU_HasMinimumFloor)
{
    // Even with tiny model, should return >= 768 MB
    size_t bytes = WorkspaceMemoryEstimator::estimate(
        1, 128, 64, 256, 100, DeviceId::cuda(0));

    EXPECT_GE(bytes, 768ULL * 1024 * 1024);
}

TEST(Test__WorkspaceMemoryEstimator, LargeVocab_ExceedsFloor)
{
    // With d_model=4096 and vocab=151936, embed_temp alone is ~2.5 GB,
    // which exceeds the 768 MB floor after the 1.1x safety margin.
    size_t bytes = WorkspaceMemoryEstimator::estimate(
        1, 4096, 4096, 16384, 151936, DeviceId::cuda(0));

    EXPECT_GT(bytes, 768ULL * 1024 * 1024);
}

TEST(Test__WorkspaceMemoryEstimator, ROCm_SameAsGPU)
{
    size_t cuda_bytes = WorkspaceMemoryEstimator::estimate(
        1, 4096, 896, 4864, 151936, DeviceId::cuda(0));
    size_t rocm_bytes = WorkspaceMemoryEstimator::estimate(
        1, 4096, 896, 4864, 151936, DeviceId::rocm(0));

    EXPECT_EQ(cuda_bytes, rocm_bytes);
}
