/**
 * @file Test__KVCacheConfigEstimate.cpp
 * @brief Tests for KVCacheConfig::estimateBytes() — the bridge between
 *        KernelFactory's KVCacheConfig and the planning estimators.
 *
 * Verifies that KVCacheConfig::estimateBytes() correctly delegates to
 * KVCacheMemoryEstimator and handles sharding, precision mapping, and
 * edge cases.
 */

#include <gtest/gtest.h>
#include "kernels/KernelFactory.h"
#include "planning/KVCacheMemoryEstimator.h"
#include "backends/DeviceId.h"

using namespace llaminar2;

namespace
{

/// Build a typical KVCacheConfig for testing.
llaminar::v2::kernels::KVCacheConfig makeConfig(
    ActivationPrecision precision = ActivationPrecision::FP16,
    int num_layers = 24,
    int max_seq_len = 4096,
    int n_kv_heads = 2,
    int head_dim = 64,
    DeviceId device = DeviceId::cuda(0),
    int local_n_kv_heads = 0)
{
    llaminar::v2::kernels::KVCacheConfig cfg;
    cfg.precision = precision;
    cfg.num_layers = num_layers;
    cfg.max_seq_len = max_seq_len;
    cfg.n_kv_heads = n_kv_heads;
    cfg.head_dim = head_dim;
    cfg.device = device;
    cfg.batch_size = 1;
    cfg.local_n_kv_heads = local_n_kv_heads;
    return cfg;
}

} // anonymous namespace

TEST(Test__KVCacheConfigEstimate, FP16_MatchesEstimator)
{
    auto cfg = makeConfig(ActivationPrecision::FP16, 24, 4096, 2, 64);
    size_t est_bytes = cfg.estimateBytes();

    size_t expected = KVCacheMemoryEstimator::estimate(
        24, 1, 4096, 2, 64, "fp16", DeviceId::cuda(0));

    EXPECT_EQ(est_bytes, expected);
    EXPECT_GT(est_bytes, 0u);
}

TEST(Test__KVCacheConfigEstimate, FP32_MatchesEstimator)
{
    auto cfg = makeConfig(ActivationPrecision::FP32, 24, 4096, 2, 64);
    size_t est_bytes = cfg.estimateBytes();

    size_t expected = KVCacheMemoryEstimator::estimate(
        24, 1, 4096, 2, 64, "fp32", DeviceId::cuda(0));

    EXPECT_EQ(est_bytes, expected);
}

TEST(Test__KVCacheConfigEstimate, Q8_1_MatchesEstimator)
{
    auto cfg = makeConfig(ActivationPrecision::Q8_1, 24, 4096, 2, 64);
    size_t est_bytes = cfg.estimateBytes();

    size_t expected = KVCacheMemoryEstimator::estimate(
        24, 1, 4096, 2, 64, "q8_1", DeviceId::cuda(0));

    EXPECT_EQ(est_bytes, expected);
}

TEST(Test__KVCacheConfigEstimate, FP32_DoubleFP16)
{
    auto cfg_fp16 = makeConfig(ActivationPrecision::FP16);
    auto cfg_fp32 = makeConfig(ActivationPrecision::FP32);

    EXPECT_EQ(cfg_fp32.estimateBytes(), cfg_fp16.estimateBytes() * 2);
}

TEST(Test__KVCacheConfigEstimate, Sharded_UsesLocalKVHeads)
{
    // Full: 2 KV heads
    auto cfg_full = makeConfig(ActivationPrecision::FP16, 24, 4096, 2, 64);

    // Sharded: 1 local KV head (TP-2)
    auto cfg_shard = makeConfig(ActivationPrecision::FP16, 24, 4096, 2, 64,
                                DeviceId::cuda(0), /*local_n_kv_heads=*/1);

    EXPECT_EQ(cfg_shard.estimateBytes(), cfg_full.estimateBytes() / 2);
}

TEST(Test__KVCacheConfigEstimate, ZeroLayers_ReturnsZero)
{
    auto cfg = makeConfig(ActivationPrecision::FP16, /*num_layers=*/0);
    EXPECT_EQ(cfg.estimateBytes(), 0u);
}

TEST(Test__KVCacheConfigEstimate, ZeroKVHeads_ReturnsZero)
{
    auto cfg = makeConfig(ActivationPrecision::FP16, 24, 4096, /*n_kv_heads=*/0);
    EXPECT_EQ(cfg.estimateBytes(), 0u);
}

TEST(Test__KVCacheConfigEstimate, ZeroHeadDim_ReturnsZero)
{
    auto cfg = makeConfig(ActivationPrecision::FP16, 24, 4096, 2, /*head_dim=*/0);
    EXPECT_EQ(cfg.estimateBytes(), 0u);
}

TEST(Test__KVCacheConfigEstimate, ScalesWithLayers)
{
    auto cfg12 = makeConfig(ActivationPrecision::FP16, 12);
    auto cfg24 = makeConfig(ActivationPrecision::FP16, 24);

    EXPECT_EQ(cfg24.estimateBytes(), cfg12.estimateBytes() * 2);
}

TEST(Test__KVCacheConfigEstimate, ScalesWithSeqLen)
{
    auto cfg2k = makeConfig(ActivationPrecision::FP16, 24, 2048);
    auto cfg4k = makeConfig(ActivationPrecision::FP16, 24, 4096);

    EXPECT_EQ(cfg4k.estimateBytes(), cfg2k.estimateBytes() * 2);
}

TEST(Test__KVCacheConfigEstimate, UnknownPrecision_FallsBackToFP16)
{
    // BF16 is not explicitly mapped in estimateBytes() → falls to default "fp16"
    auto cfg_bf16 = makeConfig(ActivationPrecision::BF16);
    auto cfg_fp16 = makeConfig(ActivationPrecision::FP16);

    EXPECT_EQ(cfg_bf16.estimateBytes(), cfg_fp16.estimateBytes());
}
