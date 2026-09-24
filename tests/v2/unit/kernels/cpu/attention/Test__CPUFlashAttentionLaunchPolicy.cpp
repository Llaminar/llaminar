/**
 * @file Test__CPUFlashAttentionLaunchPolicy.cpp
 * @brief Totality and cache-footprint tests for CPU FlashAttention2 dispatch.
 *
 * These tests deliberately avoid running attention or creating an OpenMP team.
 * They authenticate the typed production policy over dense ranges of M, K/V
 * context, local head geometry, cache capacity, and positive worker counts so a
 * future tuning-table edit cannot introduce a dispatch hole.
 */

#include <gtest/gtest.h>

#include "v2/kernels/cpu/attention/CPUFlashAttentionLaunchPolicy.h"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>

namespace
{
    using namespace llaminar2;
    using namespace llaminar2::cpu::fa2_policy;

    /** Cascade Lake cache geometry used to lock in the measured policy band. */
    inline constexpr CPUFA2CacheGeometry kOneMiBPrivateL2{
        .private_l1d_bytes = 32 * 1024,
        .private_l2_bytes = 1024 * 1024,
        .shared_l3_bytes = 38ULL * 1024 * 1024,
        .cache_line_bytes = 64,
    };

    /** @return Smallest code-generation profile that can expose one runtime ISA. */
    [[nodiscard]] constexpr CPUFA2CodegenISA codegenFor(
        CPUFA2VectorISA isa) noexcept
    {
        switch (isa)
        {
        case CPUFA2VectorISA::Scalar:
            return CPUFA2CodegenISA::Scalar;
        case CPUFA2VectorISA::AVX2:
            return CPUFA2CodegenISA::AVX2;
        case CPUFA2VectorISA::AVX512:
            return CPUFA2CodegenISA::AVX512;
        case CPUFA2VectorISA::Invalid:
            return CPUFA2CodegenISA::Invalid;
        }
        return CPUFA2CodegenISA::Invalid;
    }

    /** Build exact FP32 per-head row geometry for one policy query. */
    [[nodiscard]] constexpr CPUFA2KVTileGeometry fp32TileGeometry(
        int head_dim,
        int kv_rows,
        CPUFA2CacheGeometry cache = kOneMiBPrivateL2)
    {
        return {
            .storage_pair = CPUFA2KVStoragePair::FP32,
            .vector_isa = CPUFA2VectorISA::AVX2,
            .codegen_isa = CPUFA2CodegenISA::AVX2,
            .head_dim = head_dim,
            .kv_rows = kv_rows,
            .key_head_row_bytes =
                static_cast<std::size_t>(head_dim) * sizeof(float),
            .value_head_row_bytes =
                static_cast<std::size_t>(head_dim) * sizeof(float),
            .cache = cache,
        };
    }

    /** @return Exact stored bytes for one TurboQuant head row. */
    [[nodiscard]] constexpr std::size_t turboQuantRowBytes(
        bool tq8,
        int head_dim) noexcept
    {
        return tq8 ? 8ULL + static_cast<std::size_t>(head_dim)
                   : 8ULL + static_cast<std::size_t>(head_dim) / 2ULL;
    }

    /** Build one native TurboQuant pair geometry for policy-only tests. */
    [[nodiscard]] constexpr CPUFA2KVTileGeometry turboQuantTileGeometry(
        CPUFA2KVStoragePair pair,
        CPUFA2VectorISA isa,
        int head_dim,
        CPUFA2CacheGeometry cache = kOneMiBPrivateL2)
    {
        const bool key_is_tq8 =
            pair == CPUFA2KVStoragePair::TQ8_TQ4 ||
            pair == CPUFA2KVStoragePair::TQ8_TQ8;
        const bool value_is_tq8 =
            pair == CPUFA2KVStoragePair::TQ4_TQ8 ||
            pair == CPUFA2KVStoragePair::TQ8_TQ8;
        return {
            .storage_pair = pair,
            .vector_isa = isa,
            .codegen_isa = codegenFor(isa),
            .head_dim = head_dim,
            .kv_rows = 8192,
            .key_head_row_bytes = turboQuantRowBytes(key_is_tq8, head_dim),
            .value_head_row_bytes = turboQuantRowBytes(value_is_tq8, head_dim),
            .cache = cache,
        };
    }

    /** Build exact Q8_1 per-head row geometry for one policy query. */
    [[nodiscard]] constexpr CPUFA2KVTileGeometry q8TileGeometry(
        CPUFA2VectorISA isa,
        int head_dim,
        CPUFA2CacheGeometry cache = kOneMiBPrivateL2)
    {
        const std::size_t row_bytes =
            8ULL + static_cast<std::size_t>(head_dim);
        return {
            .storage_pair = CPUFA2KVStoragePair::Q8_1,
            .vector_isa = isa,
            .codegen_isa = codegenFor(isa),
            .head_dim = head_dim,
            .kv_rows = 8192,
            .key_head_row_bytes = row_bytes,
            .value_head_row_bytes = row_bytes,
            .cache = cache,
        };
    }
} // namespace

TEST(Test__CPUFlashAttentionLaunchPolicy,
     TileWorkingSetModelsOneStreamingOperandPerPhase)
{
    const CPUFA2KVTileGeometry geometry =
        fp32TileGeometry(/*head_dim=*/128, /*kv_rows=*/4096);

    const std::size_t expected =
        2ULL * 128 * sizeof(float) + // resident Q and output vectors
        128ULL * sizeof(float) +     // online-softmax scores
        128ULL * 128 * sizeof(float) + // one streaming K or V phase
        2ULL * 64;                   // loop metadata/cache-line allowance
    EXPECT_EQ(cpuFA2TileWorkingSetBytes(geometry, 128), expected);
}

TEST(Test__CPUFlashAttentionLaunchPolicy,
     MeasuredOneMiBL2PolicySelectsOneBatchInvariantTilePerGeometry)
{
    EXPECT_EQ(selectCPUFA2KVTile(
                  fp32TileGeometry(64, 4096)),
              256);
    EXPECT_EQ(selectCPUFA2KVTile(
                  fp32TileGeometry(128, 4096)),
              256);
    EXPECT_EQ(selectCPUFA2KVTile(
                  fp32TileGeometry(256, 4096)),
              128);
}

TEST(Test__CPUFlashAttentionLaunchPolicy,
     EmpiricalNativeCodecBandsAreCodecAndISASpecific)
{
    struct ExpectedBand
    {
        CPUFA2KVStoragePair pair = CPUFA2KVStoragePair::Invalid;
        CPUFA2VectorISA isa = CPUFA2VectorISA::Invalid;
        std::array<int, 3> tiles{};
    };
    constexpr std::array<int, 3> head_dims{64, 128, 256};
    constexpr std::array<ExpectedBand, 8> expected{{
        {CPUFA2KVStoragePair::TQ4_TQ4, CPUFA2VectorISA::AVX2, {128, 64, 128}},
        {CPUFA2KVStoragePair::TQ4_TQ8, CPUFA2VectorISA::AVX2, {256, 256, 32}},
        {CPUFA2KVStoragePair::TQ8_TQ4, CPUFA2VectorISA::AVX2, {128, 128, 128}},
        {CPUFA2KVStoragePair::TQ8_TQ8, CPUFA2VectorISA::AVX2, {256, 256, 64}},
        {CPUFA2KVStoragePair::TQ4_TQ4, CPUFA2VectorISA::AVX512, {256, 64, 128}},
        {CPUFA2KVStoragePair::TQ4_TQ8, CPUFA2VectorISA::AVX512, {256, 256, 64}},
        {CPUFA2KVStoragePair::TQ8_TQ4, CPUFA2VectorISA::AVX512, {256, 128, 64}},
        {CPUFA2KVStoragePair::TQ8_TQ8, CPUFA2VectorISA::AVX512, {128, 128, 128}},
    }};

    for (const ExpectedBand &band : expected)
    {
        for (std::size_t index = 0; index < head_dims.size(); ++index)
        {
            EXPECT_EQ(
                selectCPUFA2KVTile(turboQuantTileGeometry(
                    band.pair,
                    band.isa,
                    head_dims[index])),
                band.tiles[index])
                << cpuFA2KVStoragePairName(band.pair) << ' '
                << cpuFA2VectorISAName(band.isa)
                << " head_dim=" << head_dims[index];
        }
    }

    constexpr std::array<int, 3> q8_tiles{256, 256, 256};
    for (std::size_t index = 0; index < head_dims.size(); ++index)
    {
        EXPECT_EQ(
            selectCPUFA2KVTile(q8TileGeometry(
                CPUFA2VectorISA::AVX512,
                head_dims[index])),
            q8_tiles[index])
            << "q8_1 avx512 head_dim=" << head_dims[index];
    }

    CPUFA2KVTileGeometry fp32_avx512 = fp32TileGeometry(256, 8192);
    fp32_avx512.vector_isa = CPUFA2VectorISA::AVX512;
    fp32_avx512.codegen_isa = CPUFA2CodegenISA::AVX512;
    EXPECT_EQ(selectCPUFA2KVTile(fp32_avx512), 256);
}

TEST(Test__CPUFlashAttentionLaunchPolicy,
     AVX512CodegenWithAVX2RuntimeUsesItsOwnMeasuredBands)
{
    struct ExpectedBand
    {
        CPUFA2KVStoragePair pair = CPUFA2KVStoragePair::Invalid;
        std::array<int, 3> tiles{};
    };
    constexpr std::array<int, 3> head_dims{64, 128, 256};
    constexpr std::array<ExpectedBand, 4> expected{{
        {CPUFA2KVStoragePair::TQ4_TQ4, {128, 64, 128}},
        {CPUFA2KVStoragePair::TQ4_TQ8, {256, 256, 32}},
        {CPUFA2KVStoragePair::TQ8_TQ4, {128, 16, 32}},
        {CPUFA2KVStoragePair::TQ8_TQ8, {4, 256, 64}},
    }};

    for (const ExpectedBand &band : expected)
    {
        for (std::size_t index = 0; index < head_dims.size(); ++index)
        {
            CPUFA2KVTileGeometry geometry = turboQuantTileGeometry(
                band.pair,
                CPUFA2VectorISA::AVX2,
                head_dims[index]);
            geometry.codegen_isa = CPUFA2CodegenISA::AVX512;
            EXPECT_EQ(selectCPUFA2KVTile(geometry), band.tiles[index])
                << cpuFA2KVStoragePairName(band.pair)
                << " head_dim=" << head_dims[index];
        }
    }

    CPUFA2KVTileGeometry q16 = fp32TileGeometry(64, 8192);
    q16.storage_pair = CPUFA2KVStoragePair::Q16_1;
    q16.codegen_isa = CPUFA2CodegenISA::AVX512;
    q16.key_head_row_bytes = 136;
    q16.value_head_row_bytes = 136;
    EXPECT_EQ(selectCPUFA2KVTile(q16), 32);

    CPUFA2KVTileGeometry native_avx2 = turboQuantTileGeometry(
        CPUFA2KVStoragePair::TQ8_TQ8,
        CPUFA2VectorISA::AVX2,
        64);
    EXPECT_EQ(selectCPUFA2KVTile(native_avx2), 256);
}

TEST(Test__CPUFlashAttentionLaunchPolicy,
     EmpiricalTileNeverExceedsDetectedCacheCapacity)
{
    constexpr CPUFA2CacheGeometry tiny_private_l2{
        .private_l1d_bytes = 32 * 1024,
        .private_l2_bytes = 32 * 1024,
        .shared_l3_bytes = 8 * 1024 * 1024,
        .cache_line_bytes = 64,
    };
    EXPECT_EQ(
        selectCPUFA2KVTile(turboQuantTileGeometry(
            CPUFA2KVStoragePair::TQ8_TQ4,
            CPUFA2VectorISA::AVX2,
            256,
            tiny_private_l2)),
        8);
}

TEST(Test__CPUFlashAttentionLaunchPolicy,
     ExplicitTournamentTilesRejectUncompiledValuesInsteadOfRounding)
{
    const CPUFA2KVTileGeometry geometry =
        fp32TileGeometry(128, 4096);
    for (const int candidate : kCompiledKVTiles)
        EXPECT_EQ(selectCPUFA2KVTile(geometry, candidate), candidate);

    EXPECT_EQ(selectCPUFA2KVTile(geometry, -1), 0);
    EXPECT_EQ(selectCPUFA2KVTile(geometry, 3), 0);
    EXPECT_EQ(selectCPUFA2KVTile(geometry, 12), 0);
    EXPECT_EQ(selectCPUFA2KVTile(geometry, 2048), 0);
}

TEST(Test__CPUFlashAttentionLaunchPolicy,
     TileChoiceIsInvariantAcrossEveryPositiveKVLength)
{
    const int expected = selectCPUFA2KVTile(fp32TileGeometry(64, 1));
    ASSERT_EQ(expected, kInstalledMaximumKVTile);
    for (const int kv_rows : {1, 2, 3, 4, 5, 7, 8, 15, 16, 31, 32,
                              255, 256, 257, 8192, 131072, 1048576})
    {
        EXPECT_EQ(selectCPUFA2KVTile(fp32TileGeometry(64, kv_rows)), expected)
            << "K/V history growth must not select another reduction tree at "
            << kv_rows << " rows";
    }

    const CPUFA2KVTileGeometry geometry = fp32TileGeometry(64, 16384);
    EXPECT_EQ(selectCPUFA2KVTile(geometry, 512), 0);
    EXPECT_EQ(selectCPUFA2KVTile(geometry, 1024), 0);
}

TEST(Test__CPUFlashAttentionLaunchPolicy,
     CodecISAAndKVLengthDispatchAreTotalAndInvariant)
{
    constexpr std::array<CPUFA2KVStoragePair, 9> pairs{
        CPUFA2KVStoragePair::FP32,
        CPUFA2KVStoragePair::FP16,
        CPUFA2KVStoragePair::BF16,
        CPUFA2KVStoragePair::Q16_1,
        CPUFA2KVStoragePair::Q8_1,
        CPUFA2KVStoragePair::TQ4_TQ4,
        CPUFA2KVStoragePair::TQ4_TQ8,
        CPUFA2KVStoragePair::TQ8_TQ4,
        CPUFA2KVStoragePair::TQ8_TQ8,
    };
    constexpr std::array<CPUFA2VectorISA, 3> isas{
        CPUFA2VectorISA::Scalar,
        CPUFA2VectorISA::AVX2,
        CPUFA2VectorISA::AVX512,
    };
    constexpr std::array<CPUFA2CodegenISA, 3> codegen_isas{
        CPUFA2CodegenISA::Scalar,
        CPUFA2CodegenISA::AVX2,
        CPUFA2CodegenISA::AVX512,
    };
    constexpr std::array<int, 3> head_dims{64, 128, 256};
    constexpr std::array<int, 18> kv_rows{
        1, 2, 3, 4, 15, 16, 31, 32, 255,
        256, 257, 511, 512, 8192, 131072, 262144, 1048576, 2147483647};

    for (const CPUFA2KVStoragePair pair : pairs)
    {
        for (const CPUFA2VectorISA isa : isas)
        {
            for (const CPUFA2CodegenISA codegen_isa : codegen_isas)
            {
                if (!cpuFA2CodegenSupportsRuntimeISA(codegen_isa, isa))
                {
                    continue;
                }
                for (const int head_dim : head_dims)
                {
                    CPUFA2KVTileGeometry geometry =
                        pair >= CPUFA2KVStoragePair::TQ4_TQ4
                            ? turboQuantTileGeometry(pair, isa, head_dim)
                            : fp32TileGeometry(head_dim, 1);
                    geometry.storage_pair = pair;
                    geometry.vector_isa = isa;
                    geometry.codegen_isa = codegen_isa;
                    if (pair == CPUFA2KVStoragePair::FP16 ||
                        pair == CPUFA2KVStoragePair::BF16)
                    {
                        geometry.key_head_row_bytes =
                            static_cast<std::size_t>(head_dim) * 2;
                        geometry.value_head_row_bytes =
                            geometry.key_head_row_bytes;
                    }
                    else if (pair == CPUFA2KVStoragePair::Q16_1)
                    {
                        geometry.key_head_row_bytes =
                            static_cast<std::size_t>(head_dim) * 2 + 16;
                        geometry.value_head_row_bytes =
                            geometry.key_head_row_bytes;
                    }
                    else if (pair == CPUFA2KVStoragePair::Q8_1)
                    {
                        geometry.key_head_row_bytes =
                            static_cast<std::size_t>(head_dim) * 9 / 8;
                        geometry.value_head_row_bytes =
                            geometry.key_head_row_bytes;
                    }

                    geometry.kv_rows = kv_rows.front();
                    const int expected = selectCPUFA2KVTile(geometry);
                    ASSERT_TRUE(std::find(
                                    kCompiledKVTiles.begin(),
                                    kCompiledKVTiles.end(),
                                    expected) != kCompiledKVTiles.end())
                        << cpuFA2KVStoragePairName(pair) << ' '
                        << cpuFA2VectorISAName(isa)
                        << " codegen=" << cpuFA2CodegenISAName(codegen_isa)
                        << " head_dim=" << head_dim;
                    for (const int rows : kv_rows)
                    {
                        geometry.kv_rows = rows;
                        EXPECT_EQ(selectCPUFA2KVTile(geometry), expected)
                            << cpuFA2KVStoragePairName(pair) << ' '
                            << cpuFA2VectorISAName(isa)
                            << " codegen="
                            << cpuFA2CodegenISAName(codegen_isa)
                            << " head_dim=" << head_dim
                            << " kv_rows=" << rows;
                    }
                }
            }
        }
    }

    EXPECT_EQ(selectCPUFA2KVTile({}), 0);
    CPUFA2KVTileGeometry invalid = fp32TileGeometry(128, 4096);
    invalid.storage_pair = static_cast<CPUFA2KVStoragePair>(255);
    EXPECT_EQ(selectCPUFA2KVTile(invalid), 0);
    invalid = fp32TileGeometry(128, 4096);
    invalid.vector_isa = static_cast<CPUFA2VectorISA>(255);
    EXPECT_EQ(selectCPUFA2KVTile(invalid), 0);
    invalid = fp32TileGeometry(128, 4096);
    invalid.codegen_isa = CPUFA2CodegenISA::Invalid;
    EXPECT_EQ(selectCPUFA2KVTile(invalid), 0);
    invalid = fp32TileGeometry(128, 4096);
    invalid.vector_isa = CPUFA2VectorISA::AVX512;
    invalid.codegen_isa = CPUFA2CodegenISA::AVX2;
    EXPECT_EQ(selectCPUFA2KVTile(invalid), 0);
}

TEST(Test__CPUFlashAttentionLaunchPolicy,
     CacheAndGeometryTotalityCoversUnseenHeadContextAndStorageWidths)
{
    constexpr std::array<int, 9> head_dims{
        1, 32, 64, 80, 96, 128, 192, 256, 512};
    constexpr std::array<int, 10> kv_rows{
        1, 2, 15, 16, 255, 256, 511, 512, 131072, 1048576};
    constexpr std::array<std::size_t, 4> storage_bytes{
        1, 2, 4, 8};
    constexpr std::array<std::size_t, 5> l2_sizes{
        128 * 1024,
        256 * 1024,
        512 * 1024,
        1024 * 1024,
        2 * 1024 * 1024};

    for (const int head_dim : head_dims)
    {
        for (const int rows : kv_rows)
        {
            for (const std::size_t bytes : storage_bytes)
            {
                for (const std::size_t l2 : l2_sizes)
                {
                    const CPUFA2KVTileGeometry geometry{
                        .storage_pair = CPUFA2KVStoragePair::FP32,
                        .vector_isa = CPUFA2VectorISA::AVX2,
                        .codegen_isa = CPUFA2CodegenISA::AVX2,
                        .head_dim = head_dim,
                        .kv_rows = rows,
                        .key_head_row_bytes =
                            static_cast<std::size_t>(head_dim) * bytes,
                        .value_head_row_bytes =
                            static_cast<std::size_t>(head_dim) * bytes,
                        .cache = {
                            .private_l1d_bytes = 32 * 1024,
                            .private_l2_bytes = l2,
                            .shared_l3_bytes = 64 * 1024 * 1024,
                            .cache_line_bytes = 64,
                        },
                    };
                    const int tile = selectCPUFA2KVTile(geometry);
                    EXPECT_GE(tile, kMinimumKVTile);
                    EXPECT_LE(tile, kMaximumKVTile);
                }
            }
        }
    }
}

TEST(Test__CPUFlashAttentionLaunchPolicy,
     RequestedPhysicalAxesAreHonoredAndGeometrySelectionUsesSurplusWorkers)
{
    const CPUFA2ParallelGeometry base{
        .batch_size = 1,
        .query_rows = 1,
        .local_query_heads = 4,
        .kv_rows = 8192,
        .physical_workers = 28,
        .requested_axis =
            attention::AttentionPrefillParallelAxis::GeometrySelected,
    };

    CPUFA2ParallelPlan plan = selectCPUFA2ParallelPlan(base);
    ASSERT_TRUE(plan.valid);
    EXPECT_EQ(plan.mode, CPUFA2PhysicalMode::KeyValueContext);
    EXPECT_EQ(plan.context_partitions, 7);
    EXPECT_EQ(plan.arithmetic_partitions, 32);
    EXPECT_EQ(plan.partial_slots, 32);

    plan = selectCPUFA2ParallelPlan({
        .batch_size = 1,
        .query_rows = 128,
        .local_query_heads = 4,
        .kv_rows = 8192,
        .physical_workers = 28,
        .requested_axis =
            attention::AttentionPrefillParallelAxis::GeometrySelected,
    });
    ASSERT_TRUE(plan.valid);
    EXPECT_EQ(plan.mode, CPUFA2PhysicalMode::QuerySequence);
    EXPECT_EQ(plan.context_partitions, 1);

    plan = selectCPUFA2ParallelPlan({
        .batch_size = 1,
        .query_rows = 1,
        .local_query_heads = 4,
        .kv_rows = 8192,
        .physical_workers = 28,
        .requested_axis =
            attention::AttentionPrefillParallelAxis::QuerySequence,
    });
    ASSERT_TRUE(plan.valid);
    EXPECT_EQ(plan.mode, CPUFA2PhysicalMode::QuerySequence);

    plan = selectCPUFA2ParallelPlan({
        .batch_size = 1,
        .query_rows = 128,
        .local_query_heads = 4,
        .kv_rows = 8192,
        .physical_workers = 28,
        .requested_axis =
            attention::AttentionPrefillParallelAxis::KeyValueContext,
    });
    ASSERT_TRUE(plan.valid);
    EXPECT_EQ(plan.mode, CPUFA2PhysicalMode::KeyValueContext);
}

TEST(Test__CPUFlashAttentionLaunchPolicy,
     ContextPartitionCountIsInvariantAcrossGroupedM)
{
    int reference_partitions = 0;
    for (const int query_rows : {1, 2, 4, 8, 16})
    {
        const CPUFA2ParallelPlan plan = selectCPUFA2ParallelPlan({
            .batch_size = 1,
            .query_rows = query_rows,
            .local_query_heads = 4,
            .kv_rows = 8192,
            .physical_workers = 28,
            .requested_axis =
                attention::AttentionPrefillParallelAxis::KeyValueContext,
        });
        ASSERT_TRUE(plan.valid);
        ASSERT_TRUE(plan.usesContextParallelism());
        if (reference_partitions == 0)
            reference_partitions = plan.context_partitions;
        EXPECT_EQ(plan.context_partitions, reference_partitions);
        EXPECT_EQ(plan.arithmetic_partitions, 32);
        EXPECT_EQ(plan.context_partitions, 7);
    }
}

TEST(Test__CPUFlashAttentionLaunchPolicy,
     MContextHeadAndPositiveWorkerCountDispatchAreTotal)
{
    constexpr std::array<int, 78> query_row_samples{
        1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16,
        17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31,
        32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46,
        47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61,
        62, 63, 64, 65, 127, 128, 129, 255, 256, 257, 511, 512, 513,
        1023, 1024, 4096, 65536};
    constexpr std::array<int, 8> contexts{
        1, 16, 255, 256, 511, 512, 131072, 1048576};
    constexpr std::array<int, 9> heads{
        1, 2, 4, 7, 8, 14, 16, 32, 128};
    constexpr std::array<int, 15> worker_samples{
        1, 2, 3, 4, 7, 8, 14, 27, 28, 56, 64, 127, 128, 255, 256};
    constexpr std::array<attention::AttentionPrefillParallelAxis, 3> axes{
        attention::AttentionPrefillParallelAxis::QuerySequence,
        attention::AttentionPrefillParallelAxis::KeyValueContext,
        attention::AttentionPrefillParallelAxis::GeometrySelected};

    std::size_t checked = 0;
    for (const int query_rows : query_row_samples)
    {
        for (const int kv_rows : contexts)
        {
            for (const int local_heads : heads)
            {
                for (const int workers : worker_samples)
                {
                    for (const auto axis : axes)
                    {
                        const CPUFA2ParallelPlan plan =
                            selectCPUFA2ParallelPlan({
                                .batch_size = 1,
                                .query_rows = query_rows,
                                .local_query_heads = local_heads,
                                .kv_rows = kv_rows,
                                .physical_workers = workers,
                                .requested_axis = axis,
                            });
                        ++checked;
                        if (!plan.valid || plan.output_rows <= 0 ||
                            plan.context_partitions < 1 ||
                            plan.partial_slots < plan.output_rows ||
                            static_cast<std::int64_t>(plan.partial_slots) >
                                static_cast<std::int64_t>(plan.output_rows) *
                                    (workers + 1LL))
                        {
                            ADD_FAILURE()
                                << "dispatch hole at M=" << query_rows
                                << " K/V=" << kv_rows
                                << " heads=" << local_heads
                                << " workers=" << workers
                                << " axis=" << static_cast<int>(axis);
                            return;
                        }
                    }
                }
            }
        }
    }
    EXPECT_EQ(checked,
              query_row_samples.size() * contexts.size() * heads.size() *
                  worker_samples.size() * axes.size());
}
