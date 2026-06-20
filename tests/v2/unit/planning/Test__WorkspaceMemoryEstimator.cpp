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

TEST(Test__WorkspaceMemoryEstimator, PreparedEmbeddingPathDoesNotReserveEmbeddingTableTemp)
{
    // Qwen3.6-scale vocab and hidden dimensions should not force a
    // vocab×d_model embedding-table workspace reservation in production.
    // The embedding kernel declares that fallback buffer only when prepared
    // embedding weights are unavailable.
    size_t bytes = WorkspaceMemoryEstimator::estimate(
        1, 4096, 5120, 27648, 151936, DeviceId::cuda(0));

    EXPECT_EQ(bytes, 768ULL * 1024 * 1024);
}

TEST(Test__WorkspaceMemoryEstimator, VeryLongContextCanExceedFloorFromGemmOverhead)
{
    size_t bytes = WorkspaceMemoryEstimator::estimate(
        1, 32768, 5120, 27648, 151936, DeviceId::cuda(0));

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
