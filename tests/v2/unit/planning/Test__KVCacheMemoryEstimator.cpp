/**
 * @file Test__KVCacheMemoryEstimator.cpp
 * @brief Byte-exact tests for backend-owned persistent KV-cache allocations.
 *
 * These tests deliberately duplicate the short allocation formula of each
 * concrete cache. A change to an owner must therefore update both that owner
 * and this admission contract instead of silently underpricing capacity.
 */

#include <gtest/gtest.h>

#include "backends/DeviceId.h"
#include "planning/KVCacheMemoryEstimator.h"
#include "tensors/BlockStructures.h"

#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>

using namespace llaminar2;

namespace
{
    constexpr std::size_t kLayers = 24;
    constexpr std::size_t kBatch = 1;
    constexpr std::size_t kSequence = 4096;
    constexpr std::size_t kHeads = 2;
    constexpr std::size_t kHeadDim = 64;
    constexpr std::size_t kEntries = kLayers * kBatch;

    [[nodiscard]] constexpr std::size_t gpuMetadata(
        std::size_t pointer_tables)
    {
        return kEntries *
               (2 * sizeof(int) + pointer_tables * sizeof(void *));
    }
} // namespace

TEST(Test__KVCacheMemoryEstimator, CPUFP16OwnsOnlyKeyAndValueHorizons)
{
    const std::size_t expected =
        kEntries * 2 * kSequence * kHeads * kHeadDim * sizeof(std::uint16_t);
    EXPECT_EQ(
        KVCacheMemoryEstimator::estimate(
            kLayers, kBatch, kSequence, kHeads, kHeadDim,
            "fp16", DeviceId::cpu()),
        expected);
}

TEST(Test__KVCacheMemoryEstimator, CUDAFP16IncludesPermanentLinearizationHorizons)
{
    const std::size_t expected =
        kEntries * 4 * kSequence * kHeads * kHeadDim * sizeof(std::uint16_t) +
        gpuMetadata(2);
    EXPECT_EQ(
        KVCacheMemoryEstimator::estimate(
            kLayers, kBatch, kSequence, kHeads, kHeadDim,
            "fp16", DeviceId::cuda(0)),
        expected);
}

TEST(Test__KVCacheMemoryEstimator, ROCmFP16UsesWorkspaceScratch)
{
    const std::size_t expected =
        kEntries * 2 * kSequence * kHeads * kHeadDim * sizeof(std::uint16_t) +
        gpuMetadata(2);
    EXPECT_EQ(
        KVCacheMemoryEstimator::estimate(
            kLayers, kBatch, kSequence, kHeads, kHeadDim,
            "fp16", DeviceId::rocm(0)),
        expected);
}

TEST(Test__KVCacheMemoryEstimator, LinearFP32ChangesPayloadWithoutDoublingMetadata)
{
    const std::size_t expected =
        kEntries * 4 * kSequence * kHeads * kHeadDim * sizeof(float) +
        gpuMetadata(2);
    EXPECT_EQ(
        KVCacheMemoryEstimator::estimate(
            kLayers, kBatch, kSequence, kHeads, kHeadDim,
            "fp32", DeviceId::cuda(0)),
        expected);
}

TEST(Test__KVCacheMemoryEstimator, CPUQ8PricesPaddedPhysicalBlocks)
{
    constexpr std::size_t odd_kv_dim = 65;
    constexpr std::size_t blocks_per_row =
        (odd_kv_dim + Q8_1Block::BLOCK_SIZE - 1) /
        Q8_1Block::BLOCK_SIZE;
    const std::size_t expected =
        kEntries * 2 * kSequence * blocks_per_row * sizeof(Q8_1Block);
    EXPECT_EQ(
        KVCacheMemoryEstimator::estimate(
            kLayers, kBatch, kSequence, 1, odd_kv_dim,
            "q8_1", DeviceId::cpu()),
        expected);
    EXPECT_FLOAT_EQ(
        KVCacheMemoryEstimator::getBytesPerElement("q8_1"),
        static_cast<float>(sizeof(Q8_1Block)) /
            static_cast<float>(Q8_1Block::BLOCK_SIZE));
}

TEST(Test__KVCacheMemoryEstimator, CPUQ16UsesTheSelectedHeadBlock)
{
    const std::size_t expected =
        kEntries * 2 * kSequence * kHeads * sizeof(Q16_1Block_64);
    EXPECT_EQ(
        KVCacheMemoryEstimator::estimate(
            kLayers, kBatch, kSequence, kHeads, kHeadDim,
            "q16_1", DeviceId::cpu()),
        expected);
}

TEST(Test__KVCacheMemoryEstimator, GPUQ8PricesAQ8KeysValuesAnchorsAndTables)
{
    constexpr std::size_t value_head_bytes =
        2 * sizeof(Q8_1Block);
    const std::size_t ring_bytes =
        kEntries * kSequence * kHeads *
        (sizeof(AttentionKeyQ8Block_64) + value_head_bytes);
    const std::size_t anchor_bytes =
        kEntries * kHeads * kHeadDim * sizeof(float);
    const std::size_t expected =
        ring_bytes + anchor_bytes + gpuMetadata(3);

    EXPECT_EQ(
        KVCacheMemoryEstimator::estimate(
            kLayers, kBatch, kSequence, kHeads, kHeadDim,
            "q8_1", DeviceId::cuda(0)),
        expected);
    EXPECT_EQ(
        KVCacheMemoryEstimator::estimate(
            kLayers, kBatch, kSequence, kHeads, kHeadDim,
            "q8_1", DeviceId::rocm(0)),
        expected);
}

TEST(Test__KVCacheMemoryEstimator,
     GPULogicalPrefixBlocksMatchCUDAAndROCmNativePayloads)
{
    constexpr int tokens = 7;
    const std::size_t linear_row =
        kHeads * kHeadDim * sizeof(std::uint16_t);
    for (const DeviceId device :
         {DeviceId::cuda(0), DeviceId::rocm(0)})
    {
        const auto fp16 =
            KVCacheMemoryEstimator::estimateGPULogicalBlock(
                tokens,
                kHeads,
                kHeadDim,
                "fp16",
                device);
        EXPECT_EQ(fp16.k_bytes, tokens * linear_row);
        EXPECT_EQ(fp16.v_bytes, tokens * linear_row);

        const auto q8 =
            KVCacheMemoryEstimator::estimateGPULogicalBlock(
                tokens,
                kHeads,
                kHeadDim,
                "q8_1",
                device);
        EXPECT_EQ(
            q8.k_bytes,
            kHeads * kHeadDim * sizeof(float) +
                tokens * kHeads * sizeof(AttentionKeyQ8Block_64));
        EXPECT_EQ(
            q8.v_bytes,
            tokens * kHeads * 2 * sizeof(Q8_1Block));

        const auto tq4 =
            KVCacheMemoryEstimator::estimateGPULogicalBlock(
                tokens,
                kHeads,
                kHeadDim,
                "tq4",
                device);
        const auto tq8 =
            KVCacheMemoryEstimator::estimateGPULogicalBlock(
                tokens,
                kHeads,
                kHeadDim,
                "tq",
                device);
        EXPECT_EQ(
            tq4.k_bytes,
            tq8.k_bytes);
        EXPECT_EQ(
            tq4.v_bytes,
            tokens * kHeads * sizeof(TQ4Block_64));
        EXPECT_EQ(
            tq8.v_bytes,
            tokens * kHeads * sizeof(TQ8Block_64));
    }
}

TEST(Test__KVCacheMemoryEstimator, CPUTQ4IsTQ8KeysAndTQ4Values)
{
    const std::size_t expected =
        kEntries * kSequence * kHeads *
        (sizeof(TQ8Block_64) + sizeof(TQ4Block_64));
    EXPECT_EQ(
        KVCacheMemoryEstimator::estimate(
            kLayers, kBatch, kSequence, kHeads, kHeadDim,
            "tq4", DeviceId::cpu()),
        expected);
}

TEST(Test__KVCacheMemoryEstimator, GPUTurboQuantIncludesBothRotationMatrices)
{
    const std::size_t anchor_bytes =
        kEntries * kHeads * kHeadDim * sizeof(float);
    const std::size_t rotation_bytes =
        kLayers * kHeads * kHeadDim * kHeadDim * 2 * sizeof(float);
    const std::size_t common = anchor_bytes + gpuMetadata(3) + rotation_bytes;
    const std::size_t tq4_expected =
        kEntries * kSequence * kHeads *
            (sizeof(AttentionKeyQ8Block_64) + sizeof(TQ4Block_64)) +
        common;
    const std::size_t tq8_expected =
        kEntries * kSequence * kHeads *
            (sizeof(AttentionKeyQ8Block_64) + sizeof(TQ8Block_64)) +
        common;

    EXPECT_EQ(
        KVCacheMemoryEstimator::estimate(
            kLayers, kBatch, kSequence, kHeads, kHeadDim,
            "tq4", DeviceId::cuda(0)),
        tq4_expected);
    EXPECT_EQ(
        KVCacheMemoryEstimator::estimate(
            kLayers, kBatch, kSequence, kHeads, kHeadDim,
            "tq", DeviceId::rocm(0)),
        tq8_expected);
    EXPECT_GT(tq8_expected, tq4_expected);
}

TEST(Test__KVCacheMemoryEstimator, ZeroGeometryOwnsNoPersistentCache)
{
    EXPECT_EQ(
        KVCacheMemoryEstimator::estimate(
            0, 1, 4096, 2, 64, "fp16", DeviceId::cuda(0)),
        0U);
    EXPECT_EQ(
        KVCacheMemoryEstimator::estimate(
            24, 0, 4096, 2, 64, "fp16", DeviceId::cuda(0)),
        0U);
}

TEST(Test__KVCacheMemoryEstimator, CPUSequencePayloadScalesExactly)
{
    const std::size_t bytes_2k = KVCacheMemoryEstimator::estimate(
        24, 1, 2048, 2, 64, "fp16", DeviceId::cpu());
    const std::size_t bytes_4k = KVCacheMemoryEstimator::estimate(
        24, 1, 4096, 2, 64, "fp16", DeviceId::cpu());
    EXPECT_EQ(bytes_4k, bytes_2k * 2);
}

TEST(Test__KVCacheMemoryEstimator, UnsupportedInputsFailClosed)
{
    EXPECT_THROW(
        KVCacheMemoryEstimator::estimate(
            1, 1, 64, 1, 64, "mystery", DeviceId::cpu()),
        std::invalid_argument);
    EXPECT_THROW(
        KVCacheMemoryEstimator::estimate(
            1, 1, 64, 1, 64, "fp16", DeviceId::invalid()),
        std::invalid_argument);
    EXPECT_THROW(
        KVCacheMemoryEstimator::estimate(
            1, 1, 64, 1, 64, "q16_1", DeviceId::cuda(0)),
        std::invalid_argument);
    EXPECT_THROW(
        KVCacheMemoryEstimator::estimate(
            1, 1, 64, 1, 96, "tq4", DeviceId::rocm(0)),
        std::invalid_argument);
    EXPECT_THROW(
        KVCacheMemoryEstimator::getBytesPerElement("q16_1"),
        std::invalid_argument);
    EXPECT_THROW(
        KVCacheMemoryEstimator::getBytesPerElement("tq4"),
        std::invalid_argument);
    EXPECT_THROW(
        KVCacheMemoryEstimator::estimateGPULogicalBlock(
            64, 1, 64, "fp16", DeviceId::cpu()),
        std::invalid_argument);
    EXPECT_THROW(
        KVCacheMemoryEstimator::estimateGPULogicalBlock(
            0, 1, 64, "fp16", DeviceId::cuda(0)),
        std::invalid_argument);
    EXPECT_THROW(
        KVCacheMemoryEstimator::estimateGPULogicalBlock(
            64, 1, 64, "q16_1", DeviceId::rocm(0)),
        std::invalid_argument);
}

TEST(Test__KVCacheMemoryEstimator, ImpossibleGeometryFailsOnOverflow)
{
    constexpr int huge = std::numeric_limits<int>::max();
    EXPECT_THROW(
        KVCacheMemoryEstimator::estimate(
            huge, huge, huge, huge, huge,
            "fp32", DeviceId::cpu()),
        std::overflow_error);
}
