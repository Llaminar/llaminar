#include <gtest/gtest.h>
#include "planning/KVCacheMemoryEstimator.h"
#include "backends/DeviceId.h"

using namespace llaminar2;

TEST(Test__KVCacheMemoryEstimator, FP16_BasicFormula)
{
    // 24 layers, batch=1, seq=4096, 2 kv_heads, head_dim=64
    size_t bytes = KVCacheMemoryEstimator::estimate(
        24, 1, 4096, 2, 64, "fp16", DeviceId::cuda(0));

    // Expected: 2 (K+V) × 24 × 1 × 4096 × 2 × 64 × 2 bytes = 50,331,648 = ~48 MB
    size_t expected = 2ULL * 24 * 1 * 4096 * 2 * 64 * 2;
    EXPECT_EQ(bytes, expected);
}

TEST(Test__KVCacheMemoryEstimator, FP32_BasicFormula)
{
    size_t bytes = KVCacheMemoryEstimator::estimate(
        24, 1, 4096, 2, 64, "fp32", DeviceId::cuda(0));

    // FP32: double the FP16 bytes
    size_t fp16_bytes = KVCacheMemoryEstimator::estimate(
        24, 1, 4096, 2, 64, "fp16", DeviceId::cuda(0));
    EXPECT_EQ(bytes, fp16_bytes * 2);
}

TEST(Test__KVCacheMemoryEstimator, Q8_1_HasBlockOverhead)
{
    float bpe = KVCacheMemoryEstimator::getBytesPerElement("q8_1");
    // Q8_1 block: 36 bytes / 32 elements = 1.125
    EXPECT_NEAR(bpe, 1.125f, 0.001f);
}

TEST(Test__KVCacheMemoryEstimator, TQ_IncludesScratch)
{
    // TQ on GPU has additional scratch overhead
    size_t tq_bytes = KVCacheMemoryEstimator::estimate(
        24, 1, 4096, 2, 64, "tq", DeviceId::cuda(0));

    size_t fp16_bytes = KVCacheMemoryEstimator::estimate(
        24, 1, 4096, 2, 64, "fp16", DeviceId::cuda(0));

    // TQ should be more than FP16 due to scratch
    EXPECT_GT(tq_bytes, fp16_bytes);
}

TEST(Test__KVCacheMemoryEstimator, ZeroLayers_ReturnsZero)
{
    EXPECT_EQ(KVCacheMemoryEstimator::estimate(0, 1, 4096, 2, 64, "fp16", DeviceId::cuda(0)), 0u);
}

TEST(Test__KVCacheMemoryEstimator, ZeroBatchSize_ReturnsZero)
{
    EXPECT_EQ(KVCacheMemoryEstimator::estimate(24, 0, 4096, 2, 64, "fp16", DeviceId::cuda(0)), 0u);
}

TEST(Test__KVCacheMemoryEstimator, ScalesWithLayers)
{
    size_t bytes_12 = KVCacheMemoryEstimator::estimate(12, 1, 4096, 2, 64, "fp16", DeviceId::cuda(0));
    size_t bytes_24 = KVCacheMemoryEstimator::estimate(24, 1, 4096, 2, 64, "fp16", DeviceId::cuda(0));

    EXPECT_EQ(bytes_24, bytes_12 * 2);
}

TEST(Test__KVCacheMemoryEstimator, ScalesWithSeqLen)
{
    size_t bytes_2k = KVCacheMemoryEstimator::estimate(24, 1, 2048, 2, 64, "fp16", DeviceId::cuda(0));
    size_t bytes_4k = KVCacheMemoryEstimator::estimate(24, 1, 4096, 2, 64, "fp16", DeviceId::cuda(0));

    EXPECT_EQ(bytes_4k, bytes_2k * 2);
}
