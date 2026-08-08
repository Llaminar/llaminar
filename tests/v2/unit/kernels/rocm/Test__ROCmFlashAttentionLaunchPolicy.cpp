/**
 * @file Test__ROCmFlashAttentionLaunchPolicy.cpp
 * @brief Device-free totality tests for ROCm FA2 capture-time launch policy.
 *
 * The suite covers released Qwen dense and MoE attention geometries, equal TP
 * shards through TP8, unseen positive M values, and K/V capacities through the
 * 256K context regime. It performs no HIP runtime call or accelerator work.
 */

#include <gtest/gtest.h>

#include "kernels/rocm/attention/ROCmFlashAttentionLaunchPolicy.h"

#include <array>
#include <cstddef>
#include <limits>
#include <string>
#include <string_view>

namespace
{
    using llaminar2::attention::AttentionPrefillParallelAxis;
    using llaminar2::rocm::fa2_policy::ROCmFA2PrefillParallelGeometry;
    using llaminar2::rocm::fa2_policy::ROCmFA2PrefillPhysicalMode;
    using llaminar2::rocm::fa2_policy::kROCmFA2CanonicalContextPartitionKeys;
    using llaminar2::rocm::fa2_policy::kROCmFA2LDSCapacityBytes;
    using llaminar2::rocm::fa2_policy::maximumROCmFA2ReducerWavefronts;
    using llaminar2::rocm::fa2_policy::rocmFA2LDSBytes;
    using llaminar2::rocm::fa2_policy::rocmFA2QueryGridBlocks;
    using llaminar2::rocm::fa2_policy::selectROCmFA2GeometrySelectedWorkspaceEnvelope;
    using llaminar2::rocm::fa2_policy::selectROCmFA2PrefillParallelPlan;
    using llaminar2::rocm::fa2_policy::selectROCmFA2ReducerWavefronts;
    using llaminar2::rocm::fa2_policy::selectROCmFA2Tile;

    /** One model-visible attention geometry used by the policy sweep. */
    struct QwenAttentionGeometry
    {
        std::string_view name;
        int query_heads;
        int head_dim;
    };

    constexpr std::array kQwenAttentionGeometries{
        QwenAttentionGeometry{"Qwen2.5-0.5B", 14, 64},
        QwenAttentionGeometry{"Qwen2.5-1.5B", 12, 128},
        QwenAttentionGeometry{"Qwen2.5-3B", 16, 128},
        QwenAttentionGeometry{"Qwen2.5-7B", 28, 128},
        QwenAttentionGeometry{"Qwen2.5-14B", 40, 128},
        QwenAttentionGeometry{"Qwen3.5-0.8B", 8, 256},
        QwenAttentionGeometry{"Qwen3.5-2B", 8, 256},
        QwenAttentionGeometry{"Qwen3.5-4B", 16, 256},
        QwenAttentionGeometry{"Qwen3.5-9B", 16, 256},
        QwenAttentionGeometry{"Qwen3.5/3.6-27B", 24, 256},
        QwenAttentionGeometry{"Qwen3.5/3.6-35B-A3B", 16, 256},
        QwenAttentionGeometry{"Qwen3.5-122B-A10B", 32, 256},
        QwenAttentionGeometry{"Qwen3.5-397B-A17B", 32, 256},
    };

    TEST(ROCmFlashAttentionLaunchPolicy, SelectsOnlyCompiledLDSResidentTiles)
    {
        for (const int head_dim : {64, 128, 256})
        {
            const auto tile = selectROCmFA2Tile(head_dim);
            SCOPED_TRACE("head_dim=" + std::to_string(head_dim));
            ASSERT_TRUE(tile.valid);
            EXPECT_GT(tile.query_rows, 0);
            EXPECT_GT(tile.kv_rows, 0);
            EXPECT_EQ(
                kROCmFA2CanonicalContextPartitionKeys % tile.kv_rows,
                0);
            EXPECT_EQ(
                tile.lds_bytes,
                rocmFA2LDSBytes(
                    head_dim,
                    tile.query_rows,
                    tile.kv_rows));
            EXPECT_LE(tile.lds_bytes, kROCmFA2LDSCapacityBytes);
        }

        EXPECT_FALSE(selectROCmFA2Tile(0).valid);
        EXPECT_FALSE(selectROCmFA2Tile(257).valid);
        EXPECT_FALSE(selectROCmFA2Tile(128, 1).valid);
        EXPECT_FALSE(selectROCmFA2Tile(128, kROCmFA2LDSCapacityBytes, 7).valid);
    }

    TEST(ROCmFlashAttentionLaunchPolicy, IsTotalAcrossQwenTPAndWorkGeometry)
    {
        constexpr std::array tp_degrees{1, 2, 4, 8};
        constexpr std::array query_rows{
            1, 2, 15, 16, 17, 31, 32, 64, 127, 128, 511, 512,
            1024, 4096, 16384};
        constexpr std::array kv_capacities{
            1, 255, 256, 257, 4096, 32768, 131072, 262144};

        for (const auto &model : kQwenAttentionGeometries)
        {
            for (const int tp : tp_degrees)
            {
                if (model.query_heads % tp != 0)
                    continue;
                for (const int rows : query_rows)
                {
                    for (const int capacity : kv_capacities)
                    {
                        SCOPED_TRACE(
                            std::string(model.name) + " TP=" +
                            std::to_string(tp) + " M=" +
                            std::to_string(rows) + " KVC=" +
                            std::to_string(capacity));
                        const ROCmFA2PrefillParallelGeometry geometry{
                            .batch_size = 1,
                            .query_rows = rows,
                            .local_query_heads = model.query_heads / tp,
                            .head_dim = model.head_dim,
                            .kv_capacity = capacity,
                            .compute_unit_count = 60,
                            .lds_capacity_bytes = kROCmFA2LDSCapacityBytes,
                            .requested_axis =
                                AttentionPrefillParallelAxis::GeometrySelected,
                        };
                        const auto plan =
                            selectROCmFA2PrefillParallelPlan(geometry);
                        ASSERT_TRUE(plan.valid);
                        ASSERT_TRUE(plan.tile.valid);
                        EXPECT_GT(
                            rocmFA2QueryGridBlocks(geometry, plan.tile),
                            0);
                        EXPECT_GT(plan.query_grid_blocks, 0);
                        if (plan.usesContextParallelism())
                        {
                            EXPECT_GT(plan.context_partition_slots, 0);
                            EXPECT_LE(
                                plan.context_partition_slots,
                                plan.max_context_partitions);
                            EXPECT_GT(plan.context_phase_block_slots, 0);
                            EXPECT_LE(
                                plan.context_phase_block_slots,
                                rocmFA2ContextPhaseLogicalBlocks(
                                    geometry,
                                    plan.tile,
                                    plan.context_partition_slots));
                            EXPECT_EQ(
                                plan.device_direct_partition_limit,
                                selectROCmFA2DeviceDirectPartitionLimit(
                                    geometry,
                                    plan.tile,
                                    plan.max_context_partitions));
                            EXPECT_GT(
                                plan.reducer_dimension_wavefronts,
                                0);
                            EXPECT_GT(plan.reducer_block_slots, 0);
                            EXPECT_LE(
                                plan.reducer_block_slots,
                                rocmFA2ReducerLogicalBlocks(
                                    geometry,
                                    plan.reducer_dimension_wavefronts));
                            EXPECT_GT(plan.partial_output_bytes, 0U);
                            EXPECT_GT(plan.partial_m_bytes, 0U);
                            EXPECT_EQ(
                                plan.partial_m_bytes,
                                plan.partial_l_bytes);
                        }
                        else
                        {
                            EXPECT_EQ(plan.partial_output_bytes, 0U);
                            EXPECT_EQ(plan.partial_m_bytes, 0U);
                            EXPECT_EQ(plan.partial_l_bytes, 0U);
                            EXPECT_EQ(plan.context_phase_block_slots, 0);
                            EXPECT_EQ(plan.reducer_block_slots, 0);
                        }
                    }
                }
            }
        }
    }

    /**
     * @brief Keep packed production grids direct and explicit context bounded.
     *
     * Qwen2.5-7B TP1 at M=128 launches 56 query blocks on the 60-CU
     * reference MI50 and the static HD128 kernel wins across measured live
     * lengths, so geometry selection retains the one-node query transaction.
     * An explicitly requested context transaction remains total and uses its
     * 32-partition device-direct boundary. Qwen3-27B TP4 launches only 48 HD256
     * blocks, where immediate K/V partition parallelism is economical.
     */
    TEST(
        ROCmFlashAttentionLaunchPolicy,
        DeviceDirectSpanRequiresNearFullSingleCUWave)
    {
        const auto select = [](int local_heads)
        {
            return selectROCmFA2PrefillParallelPlan({
                .batch_size = 1,
                .query_rows = 128,
                .local_query_heads = local_heads,
                .head_dim = local_heads == 28 ? 128 : 256,
                .kv_capacity = 131072,
                .compute_unit_count = 60,
                .lds_capacity_bytes = kROCmFA2LDSCapacityBytes,
                .requested_axis =
                    AttentionPrefillParallelAxis::GeometrySelected,
            });
        };

        const auto qwen7_tp1 = select(28);
        ASSERT_TRUE(qwen7_tp1.valid);
        EXPECT_FALSE(qwen7_tp1.usesContextParallelism());
        EXPECT_EQ(qwen7_tp1.query_grid_blocks, 56);
        EXPECT_EQ(qwen7_tp1.device_direct_partition_limit, 0);

        const auto qwen7_explicit_context =
            selectROCmFA2PrefillParallelPlan({
                .batch_size = 1,
                .query_rows = 128,
                .local_query_heads = 28,
                .head_dim = 128,
                .kv_capacity = 131072,
                .compute_unit_count = 60,
                .lds_capacity_bytes = kROCmFA2LDSCapacityBytes,
                .requested_axis =
                    AttentionPrefillParallelAxis::KeyValueContext,
            });
        ASSERT_TRUE(qwen7_explicit_context.valid);
        ASSERT_TRUE(qwen7_explicit_context.usesContextParallelism());
        EXPECT_EQ(
            qwen7_explicit_context.device_direct_partition_limit,
            llaminar2::rocm::fa2_policy::
                kROCmFA2DirectMediumContextPartitions);

        const auto qwen27_tp4 = select(6);
        ASSERT_TRUE(qwen27_tp4.valid);
        ASSERT_TRUE(qwen27_tp4.usesContextParallelism());
        EXPECT_EQ(qwen27_tp4.query_grid_blocks, 48);
        EXPECT_EQ(qwen27_tp4.device_direct_partition_limit, 1);
    }

    TEST(ROCmFlashAttentionLaunchPolicy, HonorsExplicitAxesWithoutSubstitution)
    {
        const auto plan = [](AttentionPrefillParallelAxis axis)
        {
            return selectROCmFA2PrefillParallelPlan({
                .batch_size = 1,
                .query_rows = 64,
                .local_query_heads = 2,
                .head_dim = 256,
                .kv_capacity = 131072,
                .compute_unit_count = 60,
                .lds_capacity_bytes = kROCmFA2LDSCapacityBytes,
                .requested_axis = axis,
            });
        };

        const auto query = plan(AttentionPrefillParallelAxis::QuerySequence);
        ASSERT_TRUE(query.valid);
        EXPECT_EQ(query.mode, ROCmFA2PrefillPhysicalMode::QuerySequence);
        EXPECT_FALSE(query.usesContextParallelism());

        const auto context =
            plan(AttentionPrefillParallelAxis::KeyValueContext);
        ASSERT_TRUE(context.valid);
        EXPECT_EQ(context.mode, ROCmFA2PrefillPhysicalMode::KeyValueContext);
        EXPECT_TRUE(context.usesContextParallelism());
        EXPECT_EQ(context.device_direct_partition_limit, 1);
    }

    TEST(
        ROCmFlashAttentionLaunchPolicy,
        GeometrySelectionUsesOccupancyRatherThanSpeculativeDepth)
    {
        const auto plan = [](int rows, int heads, int head_dim, int capacity)
        {
            return selectROCmFA2PrefillParallelPlan({
                .batch_size = 1,
                .query_rows = rows,
                .local_query_heads = heads,
                .head_dim = head_dim,
                .kv_capacity = capacity,
                .compute_unit_count = 60,
                .requested_axis =
                    AttentionPrefillParallelAxis::GeometrySelected,
            });
        };

        for (const int rows : {1, 2, 4, 8, 16})
        {
            const auto grouped = plan(rows, 2, 256, 131072);
            SCOPED_TRACE("M=" + std::to_string(rows));
            ASSERT_TRUE(grouped.valid);
            EXPECT_TRUE(grouped.usesContextParallelism());
        }

        const auto saturated = plan(4096, 40, 128, 131072);
        ASSERT_TRUE(saturated.valid);
        EXPECT_EQ(
            saturated.mode,
            ROCmFA2PrefillPhysicalMode::QuerySequence);

        const auto single_partition = plan(1, 2, 256, 256);
        ASSERT_TRUE(single_partition.valid);
        EXPECT_EQ(
            single_partition.mode,
            ROCmFA2PrefillPhysicalMode::QuerySequence);
    }

    TEST(ROCmFlashAttentionLaunchPolicy, ContextPlanOwnsExactArenaCapacity)
    {
        const auto plan = selectROCmFA2PrefillParallelPlan({
            .batch_size = 1,
            .query_rows = 64,
            .local_query_heads = 2,
            .head_dim = 256,
            .kv_capacity = 131072,
            .compute_unit_count = 60,
            .requested_axis =
                AttentionPrefillParallelAxis::KeyValueContext,
        });
        ASSERT_TRUE(plan.valid);
        ASSERT_TRUE(plan.usesContextParallelism());
        ASSERT_EQ(plan.max_context_partitions, 512);
        EXPECT_EQ(plan.context_phase_block_slots, 60);

        constexpr std::size_t scalar_count = 64ULL * 2ULL * 512ULL;
        EXPECT_EQ(
            plan.partial_output_bytes,
            scalar_count * 256ULL * sizeof(float));
        EXPECT_EQ(plan.partial_m_bytes, scalar_count * sizeof(float));
        EXPECT_EQ(plan.partial_l_bytes, scalar_count * sizeof(float));
    }

    /**
     * @brief Prove family workspace follows the intermediate-M mode maximum.
     *
     * Qwen3.6-35B at M=4096 has enough query tiles to select the direct plan,
     * but its M=464 participant is the last grid below the eight-CU-wave
     * context threshold. The family envelope must reserve that participant's
     * summaries before either graph captures the shared arena address.
     */
    TEST(
        ROCmFlashAttentionLaunchPolicy,
        GeometrySelectedWorkspaceEnvelopeCoversNonMonotonicM)
    {
        const ROCmFA2PrefillParallelGeometry family_geometry{
            .batch_size = 1,
            .query_rows = 4096,
            .local_query_heads = 16,
            .head_dim = 256,
            .kv_capacity = 4096,
            .compute_unit_count = 60,
            .lds_capacity_bytes = kROCmFA2LDSCapacityBytes,
            .requested_axis =
                AttentionPrefillParallelAxis::GeometrySelected,
        };

        const auto largest_participant =
            selectROCmFA2PrefillParallelPlan(family_geometry);
        ASSERT_TRUE(largest_participant.valid);
        EXPECT_FALSE(largest_participant.usesContextParallelism());
        EXPECT_EQ(largest_participant.partial_output_bytes, 0U);

        const auto envelope =
            selectROCmFA2GeometrySelectedWorkspaceEnvelope(
                family_geometry);
        ASSERT_TRUE(envelope.valid);
        ASSERT_TRUE(envelope.usesContextParallelism());
        EXPECT_EQ(envelope.query_grid_blocks, 464);
        EXPECT_EQ(envelope.max_context_partitions, 16);

        constexpr std::size_t scalar_count =
            464ULL * 16ULL * 16ULL;
        EXPECT_EQ(
            envelope.partial_output_bytes,
            scalar_count * 256ULL * sizeof(float));
        EXPECT_EQ(
            envelope.partial_m_bytes,
            scalar_count * sizeof(float));
        EXPECT_EQ(envelope.partial_l_bytes, envelope.partial_m_bytes);
    }

    /**
     * @brief Prove positive context capacities have no dispatch or size holes.
     *
     * Timing evidence stops at 128K, but capture planning is integer arithmetic
     * and must remain total beyond measured points. The test exhausts every
     * capacity through one million, then crosses the final positive `int`
     * boundaries. The explicit 256K assertion authenticates the exact arena
     * bytes that production preflight must publish.
     */
    TEST(
        ROCmFlashAttentionLaunchPolicy,
        PositiveContextCapacityTotalityIncludesExact256KWorkspace)
    {
        for (int capacity = 1; capacity <= 1048576; ++capacity)
        {
            const ROCmFA2PrefillParallelGeometry geometry{
                .batch_size = 1,
                .query_rows = capacity,
                .local_query_heads = 16,
                .head_dim = 256,
                .kv_capacity = capacity,
                .compute_unit_count = 60,
                .lds_capacity_bytes = kROCmFA2LDSCapacityBytes,
                .requested_axis =
                    AttentionPrefillParallelAxis::GeometrySelected,
            };
            const auto executable =
                selectROCmFA2PrefillParallelPlan(geometry);
            const auto workspace =
                selectROCmFA2GeometrySelectedWorkspaceEnvelope(geometry);
            const bool workspace_is_total =
                capacity <= kROCmFA2CanonicalContextPartitionKeys
                    ? !workspace.valid
                    : workspace.valid &&
                          workspace.usesContextParallelism();
            if (!executable.valid || executable.query_grid_blocks <= 0 ||
                !workspace_is_total)
            {
                ADD_FAILURE()
                    << "ROCm FA2 policy has a positive-capacity hole at "
                    << capacity;
                break;
            }
        }

        constexpr std::array capacities{
            std::numeric_limits<int>::max() - 255,
            std::numeric_limits<int>::max(),
        };

        for (const int capacity : capacities)
        {
            SCOPED_TRACE("capacity=" + std::to_string(capacity));
            const ROCmFA2PrefillParallelGeometry geometry{
                .batch_size = 1,
                .query_rows = capacity,
                .local_query_heads = 16,
                .head_dim = 256,
                .kv_capacity = capacity,
                .compute_unit_count = 60,
                .lds_capacity_bytes = kROCmFA2LDSCapacityBytes,
                .requested_axis =
                    AttentionPrefillParallelAxis::GeometrySelected,
            };
            const auto executable =
                selectROCmFA2PrefillParallelPlan(geometry);
            ASSERT_TRUE(executable.valid);
            EXPECT_GT(executable.query_grid_blocks, 0);

            const auto workspace =
                selectROCmFA2GeometrySelectedWorkspaceEnvelope(
                    geometry);
            if (capacity <= kROCmFA2CanonicalContextPartitionKeys)
            {
                EXPECT_FALSE(workspace.valid);
                continue;
            }

            ASSERT_TRUE(workspace.valid);
            ASSERT_TRUE(workspace.usesContextParallelism());
            EXPECT_EQ(
                workspace.max_context_partitions,
                static_cast<int>(
                    (static_cast<std::int64_t>(capacity) +
                     kROCmFA2CanonicalContextPartitionKeys - 1) /
                    kROCmFA2CanonicalContextPartitionKeys));
            EXPECT_GT(workspace.partial_output_bytes, 0U);
            EXPECT_GT(workspace.partial_m_bytes, 0U);
            EXPECT_EQ(
                workspace.partial_m_bytes,
                workspace.partial_l_bytes);
        }

        constexpr std::size_t kRows = 464;
        constexpr std::size_t kHeads = 16;
        constexpr std::size_t kPartitions = 1024;
        constexpr std::size_t kHeadDim = 256;
        const auto context_256k =
            selectROCmFA2GeometrySelectedWorkspaceEnvelope({
                .batch_size = 1,
                .query_rows = 262144,
                .local_query_heads = static_cast<int>(kHeads),
                .head_dim = static_cast<int>(kHeadDim),
                .kv_capacity = 262144,
                .compute_unit_count = 60,
                .lds_capacity_bytes = kROCmFA2LDSCapacityBytes,
                .requested_axis =
                    AttentionPrefillParallelAxis::GeometrySelected,
            });
        ASSERT_TRUE(context_256k.valid);
        EXPECT_EQ(context_256k.query_grid_blocks, 464);
        EXPECT_EQ(context_256k.max_context_partitions, 1024);
        EXPECT_EQ(
            context_256k.partial_output_bytes,
            kRows * kHeads * kPartitions * kHeadDim * sizeof(float));
        EXPECT_EQ(
            context_256k.partial_m_bytes,
            kRows * kHeads * kPartitions * sizeof(float));
    }

    TEST(
        ROCmFlashAttentionLaunchPolicy,
        LogicalContextOwnersCoverEveryCapacityPartition)
    {
        for (const int head_dim : {64, 128})
        {
            for (const int rows : {17, 64, 128})
            {
                const auto plan = selectROCmFA2PrefillParallelPlan({
                    .batch_size = 1,
                    .query_rows = rows,
                    .local_query_heads = 7,
                    .head_dim = head_dim,
                    .kv_capacity = 131072,
                    .compute_unit_count = 60,
                    .requested_axis =
                        AttentionPrefillParallelAxis::KeyValueContext,
                });
                SCOPED_TRACE(
                    "head_dim=" + std::to_string(head_dim) +
                    " M=" + std::to_string(rows));
                ASSERT_TRUE(plan.valid);
                ASSERT_TRUE(plan.usesContextParallelism());
                EXPECT_EQ(plan.max_context_partitions, 512);
                EXPECT_EQ(
                    plan.context_partition_slots,
                    plan.max_context_partitions);
                EXPECT_EQ(plan.context_phase_block_slots, 60);
            }
        }

        const auto wide_head_plan = selectROCmFA2PrefillParallelPlan({
            .batch_size = 1,
            .query_rows = 64,
            .local_query_heads = 2,
            .head_dim = 256,
            .kv_capacity = 131072,
            .compute_unit_count = 60,
            .requested_axis =
                AttentionPrefillParallelAxis::KeyValueContext,
        });
        ASSERT_TRUE(wide_head_plan.valid);
        EXPECT_EQ(
            wide_head_plan.context_partition_slots,
            wide_head_plan.max_context_partitions);
    }

    TEST(ROCmFlashAttentionLaunchPolicy, ReducerStripingIsTotalAndBounded)
    {
        for (const int head_dim : {64, 128, 256})
        {
            for (const int rows : {1, 17, 64, 128, 4096})
            {
                for (const int heads : {1, 2, 8, 16, 40})
                {
                    const ROCmFA2PrefillParallelGeometry geometry{
                        .batch_size = 1,
                        .query_rows = rows,
                        .local_query_heads = heads,
                        .head_dim = head_dim,
                        .kv_capacity = 131072,
                        .compute_unit_count = 60,
                    };
                    const int selected =
                        selectROCmFA2ReducerWavefronts(geometry);
                    ASSERT_GE(selected, 1);
                    ASSERT_LE(
                        selected,
                        maximumROCmFA2ReducerWavefronts(head_dim));
                    EXPECT_EQ(selected & (selected - 1), 0);
                }
            }
        }
    }

    TEST(
        ROCmFlashAttentionLaunchPolicy,
        PersistentReducerGridCoversEveryLogicalTask)
    {
        const ROCmFA2PrefillParallelGeometry geometry{
            .batch_size = 1,
            .query_rows = 128,
            .local_query_heads = 6,
            .head_dim = 256,
            .kv_capacity = 131072,
            .compute_unit_count = 60,
            .requested_axis =
                AttentionPrefillParallelAxis::KeyValueContext,
        };

        for (const int wavefronts : {1, 2, 4})
        {
            const std::int64_t logical_blocks =
                rocmFA2ReducerLogicalBlocks(geometry, wavefronts);
            ASSERT_GT(logical_blocks, 0);
            EXPECT_EQ(
                selectROCmFA2ReducerBlockSlots(geometry, wavefronts),
                logical_blocks);
            for (const int explicit_blocks : {1, 2, 4, 8, 16, 30, 60, 120})
            {
                if (explicit_blocks <= logical_blocks)
                {
                    EXPECT_EQ(
                        selectROCmFA2ReducerBlockSlots(
                            geometry,
                            wavefronts,
                            explicit_blocks),
                        explicit_blocks);
                }
            }
            EXPECT_EQ(
                selectROCmFA2ReducerBlockSlots(
                    geometry,
                    wavefronts,
                    static_cast<int>(logical_blocks) + 1),
                0);
        }

        EXPECT_EQ(selectROCmFA2ReducerBlockSlots(geometry, 3), 0);
        EXPECT_EQ(selectROCmFA2ReducerBlockSlots(geometry, 1, -1), 0);
    }

    TEST(
        ROCmFlashAttentionLaunchPolicy,
        PersistentContextPhaseGridMatchesLDSResidency)
    {
        const ROCmFA2PrefillParallelGeometry geometry{
            .batch_size = 1,
            .query_rows = 128,
            .local_query_heads = 6,
            .head_dim = 256,
            .kv_capacity = 131072,
            .compute_unit_count = 60,
            .requested_axis =
                AttentionPrefillParallelAxis::KeyValueContext,
        };
        const auto tile = selectROCmFA2Tile(geometry.head_dim);
        ASSERT_TRUE(tile.valid);
        ASSERT_GT(tile.lds_bytes, kROCmFA2LDSCapacityBytes / 2);

        constexpr int partition_slots = 10;
        EXPECT_EQ(
            rocmFA2ContextPhaseLogicalBlocks(
                geometry,
                tile,
                partition_slots),
            480);
        EXPECT_EQ(
            selectROCmFA2ContextPhaseBlockSlots(
                geometry,
                tile,
                partition_slots),
            60);
        EXPECT_EQ(
            selectROCmFA2ContextPhaseBlockSlots(
                geometry,
                tile,
                partition_slots,
                30),
            30);
        EXPECT_EQ(
            selectROCmFA2ContextPhaseBlockSlots(
                geometry,
                tile,
                partition_slots,
                481),
            0);

        for (const int head_dim : {64, 128, 256})
        {
            const auto selected = selectROCmFA2Tile(head_dim);
            ASSERT_TRUE(selected.valid);
            EXPECT_GT(
                selected.lds_bytes,
                kROCmFA2LDSCapacityBytes / 2)
                << "head_dim=" << head_dim;
        }
    }

} // namespace
