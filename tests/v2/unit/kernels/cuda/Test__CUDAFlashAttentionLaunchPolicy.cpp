/**
 * @file Test__CUDAFlashAttentionLaunchPolicy.cpp
 * @brief Device-free totality tests for CUDA FA2 capture-time launch geometry.
 *
 * These tests cover the attention geometries used by Qwen dense releases from
 * 0.5B through 27B and the 35B/122B/397B MoE releases.  They also exhaust a
 * broad synthetic geometry lattice so an unseen positive query size, local TP
 * head shard, batch size, or SM count cannot leave a dispatch hole.  No CUDA
 * runtime call or GPU work occurs in this unit suite.
 */

#include <gtest/gtest.h>

#include "kernels/cuda/attention/CUDAFlashAttentionLaunchPolicy.h"

#include <array>
#include <string>
#include <string_view>

namespace
{
    using llaminar2::cuda::fa2_policy::FA2KVTileGeometry;
    using llaminar2::cuda::fa2_policy::FA2PrefillParallelGeometry;
    using llaminar2::cuda::fa2_policy::FA2PrefillPhysicalMode;
    using llaminar2::cuda::fa2_policy::FA2QueryPartitionGeometry;
    using llaminar2::cuda::fa2_policy::FA2HeadMappingGeometry;
    using llaminar2::cuda::fa2_policy::fa2DynamicSharedMemoryBytes;
    using llaminar2::cuda::fa2_policy::fa2QueryGridBlocks;
    using llaminar2::cuda::fa2_policy::isCompiledFA2PVWarpGeometry;
    using llaminar2::cuda::fa2_policy::isValidFA2HeadMapping;
    using llaminar2::cuda::fa2_policy::kFA2ContextTargetBlockWaves;
    using llaminar2::cuda::fa2_policy::maximumFA2QueryWarpGroups;
    using llaminar2::cuda::fa2_policy::maximumFA2PVWarpsPerQueryGroup;
    using llaminar2::cuda::fa2_policy::maximumFA2ReducerDimensionWarps;
    using llaminar2::cuda::fa2_policy::preferredFA2QueryGridWaves;
    using llaminar2::cuda::fa2_policy::selectFA2QueryWarpGroups;
    using llaminar2::cuda::fa2_policy::selectFA2KVTile;
    using llaminar2::cuda::fa2_policy::selectFA2ContextPartitionSlots;
    using llaminar2::cuda::fa2_policy::selectFA2DeviceDirectPartitionLimit;
    using llaminar2::cuda::fa2_policy::selectFA2PrefillParallelPlan;
    using llaminar2::cuda::fa2_policy::selectFA2ReducerDimensionWarps;
    using llaminar2::attention::AttentionPrefillParallelAxis;

    /**
     * @brief One named release geometry used to make policy changes reviewable.
     */
    struct QwenAttentionGeometry
    {
        std::string_view name;
        int query_heads;
        int head_dim;
        int expected_groups_at_m64_on_ga102;
    };

    TEST(CUDAFlashAttentionLaunchPolicy, CoversQwenDenseAndMoEReleaseEnvelope)
    {
        constexpr std::array geometries{
            QwenAttentionGeometry{"Qwen2.5-0.5B", 14, 64, 1},
            QwenAttentionGeometry{"Qwen2.5-1.5B", 12, 128, 1},
            QwenAttentionGeometry{"Qwen2.5-3B", 16, 128, 1},
            QwenAttentionGeometry{"Qwen2.5-7B", 28, 128, 1},
            QwenAttentionGeometry{"Qwen2.5-14B", 40, 128, 1},
            QwenAttentionGeometry{"Qwen3.5-0.8B", 8, 256, 1},
            QwenAttentionGeometry{"Qwen3.5-2B", 8, 256, 1},
            QwenAttentionGeometry{"Qwen3.5-4B", 16, 256, 1},
            QwenAttentionGeometry{"Qwen3.5-9B", 16, 256, 1},
            QwenAttentionGeometry{"Qwen3.5/3.6-27B", 24, 256, 1},
            QwenAttentionGeometry{"Qwen3.5/3.6-35B-A3B", 16, 256, 1},
            QwenAttentionGeometry{"Qwen3.5-122B-A10B", 32, 256, 1},
            QwenAttentionGeometry{"Qwen3.5-397B-A17B", 32, 256, 1},
        };

        for (const auto &model : geometries)
        {
            SCOPED_TRACE(model.name);
            const FA2QueryPartitionGeometry geometry{
                .batch_size = 1,
                .query_rows = 64,
                .local_query_heads = model.query_heads,
                .head_dim = model.head_dim,
                .sm_count = 82,
            };
            EXPECT_EQ(
                selectFA2QueryWarpGroups(geometry),
                model.expected_groups_at_m64_on_ga102);
            EXPECT_LE(
                fa2QueryGridBlocks(
                    geometry,
                    model.expected_groups_at_m64_on_ga102),
                static_cast<std::int64_t>(geometry.sm_count) *
                    preferredFA2QueryGridWaves(geometry.head_dim));
        }
    }

    TEST(CUDAFlashAttentionLaunchPolicy, IsTotalAcrossPositiveGeometryLattice)
    {
        constexpr std::array head_dims{64, 128, 256};
        constexpr std::array batches{1, 2, 4};
        constexpr std::array query_rows{
            1, 2, 15, 16, 17, 32, 48, 64, 80, 96, 128, 256,
            384, 512, 1024, 2048, 4096};
        constexpr std::array local_heads{1, 2, 4, 8, 12, 14, 16, 24, 28, 32, 40, 64};
        constexpr std::array sm_counts{1, 16, 40, 60, 72, 80, 82, 108, 120, 128};

        for (const int head_dim : head_dims)
        {
            const int maximum_groups =
                maximumFA2QueryWarpGroups(head_dim);
            ASSERT_GT(maximum_groups, 0);
            for (const int batch_size : batches)
            {
                for (const int rows : query_rows)
                {
                    for (const int heads : local_heads)
                    {
                        for (const int sms : sm_counts)
                        {
                            const FA2QueryPartitionGeometry geometry{
                                .batch_size = batch_size,
                                .query_rows = rows,
                                .local_query_heads = heads,
                                .head_dim = head_dim,
                                .sm_count = sms,
                            };
                            const int selected =
                                selectFA2QueryWarpGroups(geometry);
                            ASSERT_GE(selected, 1);
                            ASSERT_LE(selected, maximum_groups);
                            ASSERT_GT(fa2QueryGridBlocks(geometry, selected), 0);

                            if (selected < maximum_groups)
                            {
                                EXPECT_LE(
                                    fa2QueryGridBlocks(geometry, selected),
                                    static_cast<std::int64_t>(sms) *
                                        preferredFA2QueryGridWaves(head_dim));
                            }
                            if (selected > 1)
                            {
                                EXPECT_GT(
                                    fa2QueryGridBlocks(geometry, selected - 1),
                                    static_cast<std::int64_t>(sms) *
                                        preferredFA2QueryGridWaves(head_dim));
                            }
                        }
                    }
                }
            }
        }
    }

    TEST(CUDAFlashAttentionLaunchPolicy, CoversEqualTensorSplitsAtTP2TP4TP8)
    {
        constexpr std::array models{
            QwenAttentionGeometry{"Qwen2.5-0.5B", 14, 64, 0},
            QwenAttentionGeometry{"Qwen2.5-1.5B", 12, 128, 0},
            QwenAttentionGeometry{"Qwen2.5-3B", 16, 128, 0},
            QwenAttentionGeometry{"Qwen2.5-7B", 28, 128, 0},
            QwenAttentionGeometry{"Qwen2.5-14B", 40, 128, 0},
            QwenAttentionGeometry{"Qwen3.5-0.8B", 8, 256, 0},
            QwenAttentionGeometry{"Qwen3.5-4B", 16, 256, 0},
            QwenAttentionGeometry{"Qwen3.5/3.6-27B", 24, 256, 0},
            QwenAttentionGeometry{"Qwen3.5/3.6-35B-A3B", 16, 256, 0},
            QwenAttentionGeometry{"Qwen3.5-122B/397B", 32, 256, 0},
        };
        constexpr std::array tp_degrees{2, 4, 8};

        for (const auto &model : models)
        {
            for (const int tp_degree : tp_degrees)
            {
                if (model.query_heads % tp_degree != 0)
                    continue;
                SCOPED_TRACE(
                    std::string(model.name) + " TP=" +
                    std::to_string(tp_degree));
                const FA2QueryPartitionGeometry geometry{
                    .batch_size = 1,
                    .query_rows = 64,
                    .local_query_heads = model.query_heads / tp_degree,
                    .head_dim = model.head_dim,
                    .sm_count = 82,
                };
                const int selected = selectFA2QueryWarpGroups(geometry);
                EXPECT_GE(selected, 1);
                EXPECT_LE(selected, maximumFA2QueryWarpGroups(model.head_dim));
                EXPECT_GT(fa2QueryGridBlocks(geometry, selected), 0);
            }
        }
    }

    TEST(CUDAFlashAttentionLaunchPolicy, ReducerDimensionPolicyFillsSmallMTPShards)
    {
        const auto selected = [](
                                  int rows,
                                  int heads,
                                  int head_dim)
        {
            return selectFA2ReducerDimensionWarps({
                .batch_size = 1,
                .query_rows = rows,
                .local_query_heads = heads,
                .head_dim = head_dim,
                .sm_count = 82,
            });
        };

        // Qwen3.6-35B TP8 remains underfilled even with every HD256 stripe.
        EXPECT_EQ(selected(/*rows=*/16, /*heads=*/2, /*head_dim=*/256), 8);
        EXPECT_EQ(selected(/*rows=*/64, /*heads=*/2, /*head_dim=*/256), 8);
        EXPECT_EQ(selected(/*rows=*/256, /*heads=*/2, /*head_dim=*/256), 4);

        // More local heads need less duplicated scalar reducer work.
        EXPECT_EQ(selected(/*rows=*/64, /*heads=*/16, /*head_dim=*/256), 2);

        // Dimension striping never creates warps with no owned dimensions.
        EXPECT_EQ(selected(/*rows=*/64, /*heads=*/2, /*head_dim=*/128), 4);
        EXPECT_EQ(selected(/*rows=*/64, /*heads=*/2, /*head_dim=*/64), 2);
    }

    TEST(
        CUDAFlashAttentionLaunchPolicy,
        GeometrySelectedCapturesDeviceAdaptiveTransaction)
    {
        const auto plan = [](
                              int query_rows,
                              int local_heads,
                              AttentionPrefillParallelAxis requested_axis =
                                  AttentionPrefillParallelAxis::GeometrySelected)
        {
            return selectFA2PrefillParallelPlan({
                .batch_size = 1,
                .query_rows = query_rows,
                .local_query_heads = local_heads,
                .head_dim = 256,
                .kv_capacity = 131072,
                .sm_count = 82,
                .requested_axis = requested_axis,
            });
        };

        // Every long-capacity geometry with a profitable context window owns
        // one immutable adaptive graph. Local head count and M change its
        // physical threshold; they never cause a host-time live-length choice.
        for (const int query_rows : {64, 128, 256})
        {
            for (const int local_heads : {2, 8, 16})
            {
                const auto adaptive = plan(query_rows, local_heads);
                ASSERT_TRUE(adaptive.valid);
                EXPECT_EQ(
                    adaptive.mode,
                    FA2PrefillPhysicalMode::DeviceAdaptive);
                EXPECT_TRUE(adaptive.usesContextParallelism());
                EXPECT_TRUE(adaptive.usesDeviceAdaptiveParallelism());
                EXPECT_GT(adaptive.device_direct_partition_limit, 0);
                EXPECT_LT(
                    adaptive.device_direct_partition_limit,
                    adaptive.max_context_partitions);
            }
        }

        // An explicit model policy is exact and never reinterpreted as advice.
        const auto query =
            plan(64, 2, AttentionPrefillParallelAxis::QuerySequence);
        ASSERT_TRUE(query.valid);
        EXPECT_EQ(query.mode, FA2PrefillPhysicalMode::QuerySequence);
        EXPECT_EQ(query.device_direct_partition_limit, 0);
        EXPECT_FALSE(query.usesContextParallelism());

        const auto context =
            plan(256, 16, AttentionPrefillParallelAxis::KeyValueContext);
        ASSERT_TRUE(context.valid);
        EXPECT_EQ(context.mode, FA2PrefillPhysicalMode::KeyValueContext);
        EXPECT_EQ(context.device_direct_partition_limit, 0);
        EXPECT_TRUE(context.usesContextParallelism());
        EXPECT_FALSE(context.usesDeviceAdaptiveParallelism());
    }

    TEST(
        CUDAFlashAttentionLaunchPolicy,
        DeviceDirectPrefixPolicyUsesPhysicalGeometry)
    {
        const auto direct_limit = [](
                                      int query_rows,
                                      int local_heads,
                                      int head_dim,
                                      int max_partitions = 512)
        {
            const FA2QueryPartitionGeometry geometry{
                .batch_size = 1,
                .query_rows = query_rows,
                .local_query_heads = local_heads,
                .head_dim = head_dim,
                .sm_count = 82,
            };
            return selectFA2DeviceDirectPartitionLimit(
                geometry,
                selectFA2QueryWarpGroups(geometry),
                max_partitions);
        };

        // Sparse HD256 grids cross after one partition. A one-to-two-wave grid
        // retains direct publication for the second partition as well.
        EXPECT_EQ(direct_limit(64, 2, 256), 1);
        EXPECT_EQ(direct_limit(64, 24, 256), 2);

        // Saturated HD128 grids retain direct execution while M is small. At
        // M=128, a 1.56-wave one-group grid benefits from context after one
        // partition, while a 1.95-wave grid is already economically saturated.
        // The saturated two-group geometry crosses after 32 partitions.
        EXPECT_EQ(direct_limit(64, 40, 128), 512);
        EXPECT_EQ(direct_limit(128, 16, 128), 1);
        EXPECT_EQ(direct_limit(128, 20, 128), 512);
        EXPECT_EQ(direct_limit(128, 40, 128), 32);
        EXPECT_EQ(direct_limit(128, 28, 128), 2);
        EXPECT_EQ(direct_limit(128, 2, 128), 1);

        EXPECT_EQ(direct_limit(128, 14, 64), 1);
        EXPECT_EQ(direct_limit(128, 40, 128, 8), 8);
        EXPECT_EQ(selectFA2DeviceDirectPartitionLimit({}, 1, 1), -1);
    }

    TEST(CUDAFlashAttentionLaunchPolicy, ContextPlanOwnsExactPersistentWorkspace)
    {
        const auto plan = selectFA2PrefillParallelPlan({
            .batch_size = 1,
            .query_rows = 64,
            .local_query_heads = 2,
            .head_dim = 256,
            .kv_capacity = 131072,
            .sm_count = 82,
            .requested_axis =
                AttentionPrefillParallelAxis::KeyValueContext,
        });
        ASSERT_TRUE(plan.valid);
        ASSERT_TRUE(plan.usesContextParallelism());
        EXPECT_EQ(plan.max_context_partitions, 512);
        EXPECT_EQ(plan.reducer_dimension_warps, 8);
        EXPECT_GT(plan.context_partition_slots, 0);
        EXPECT_LE(
            plan.context_partition_slots,
            plan.max_context_partitions);

        constexpr std::size_t scalar_count = 64ULL * 2ULL * 512ULL;
        EXPECT_EQ(
            plan.partial_output_bytes,
            scalar_count * 256ULL * sizeof(float));
        EXPECT_EQ(plan.partial_m_bytes, scalar_count * sizeof(float));
        EXPECT_EQ(plan.partial_l_bytes, scalar_count * sizeof(float));

        const auto query_plan = selectFA2PrefillParallelPlan({
            .batch_size = 1,
            .query_rows = 512,
            .local_query_heads = 16,
            .head_dim = 256,
            .kv_capacity = 131072,
            .sm_count = 82,
            .requested_axis =
                AttentionPrefillParallelAxis::QuerySequence,
        });
        ASSERT_TRUE(query_plan.valid);
        EXPECT_FALSE(query_plan.usesContextParallelism());
        EXPECT_EQ(query_plan.context_partition_slots, 0);
        EXPECT_EQ(query_plan.partial_output_bytes, 0u);
        EXPECT_EQ(query_plan.partial_m_bytes, 0u);
        EXPECT_EQ(query_plan.partial_l_bytes, 0u);
    }

    TEST(
        CUDAFlashAttentionLaunchPolicy,
        ContextPartitionSlotsReachProductiveWaveBudget)
    {
        constexpr FA2QueryPartitionGeometry geometry{
            .batch_size = 1,
            .query_rows = 64,
            .local_query_heads = 2,
            .head_dim = 256,
            .sm_count = 82,
        };
        constexpr int max_partitions = 512;
        constexpr int query_groups = selectFA2QueryWarpGroups(geometry);
        constexpr std::int64_t base_blocks =
            fa2QueryGridBlocks(geometry, query_groups);
        constexpr int expected_slots = static_cast<int>(
            (geometry.sm_count * kFA2ContextTargetBlockWaves +
             base_blocks - 1) /
            base_blocks);

        EXPECT_EQ(
            selectFA2ContextPartitionSlots(geometry, max_partitions),
            expected_slots);
        const std::int64_t target_blocks =
            static_cast<std::int64_t>(geometry.sm_count) *
            kFA2ContextTargetBlockWaves;
        EXPECT_GE(base_blocks * expected_slots, target_blocks);
        EXPECT_LT(base_blocks * (expected_slots - 1), target_blocks);

        EXPECT_EQ(
            selectFA2ContextPartitionSlots(
                geometry, max_partitions, /*explicit_slots=*/1),
            1);
        EXPECT_EQ(
            selectFA2ContextPartitionSlots(
                geometry, max_partitions, max_partitions),
            max_partitions);
        EXPECT_EQ(
            selectFA2ContextPartitionSlots(
                geometry, max_partitions, max_partitions + 1),
            0);
        EXPECT_EQ(selectFA2ContextPartitionSlots({}, max_partitions), 0);
        EXPECT_EQ(selectFA2ContextPartitionSlots(geometry, 0), 0);
    }

    TEST(CUDAFlashAttentionLaunchPolicy, PrefillPlanIsTotalAndFailsClosed)
    {
        constexpr std::array head_dims{64, 128, 256};
        constexpr std::array batches{1, 2, 4};
        constexpr std::array query_rows{
            1, 2, 16, 17, 32, 64, 128, 256, 512, 1024, 4096};
        constexpr std::array local_heads{1, 2, 4, 8, 16, 32, 64};
        constexpr std::array capacities{1, 255, 256, 257, 4096, 131072, 262144};
        constexpr std::array sm_counts{1, 16, 60, 82, 120, 128};
        constexpr std::array axes{
            AttentionPrefillParallelAxis::QuerySequence,
            AttentionPrefillParallelAxis::KeyValueContext,
            AttentionPrefillParallelAxis::GeometrySelected,
        };

        for (const int head_dim : head_dims)
        {
            for (const int batch : batches)
            {
                for (const int rows : query_rows)
                {
                    for (const int heads : local_heads)
                    {
                        for (const int capacity : capacities)
                        {
                            for (const int sms : sm_counts)
                            {
                                for (const auto axis : axes)
                                {
                                    const auto plan =
                                        selectFA2PrefillParallelPlan({
                                            .batch_size = batch,
                                            .query_rows = rows,
                                            .local_query_heads = heads,
                                            .head_dim = head_dim,
                                            .kv_capacity = capacity,
                                            .sm_count = sms,
                                            .requested_axis = axis,
                                        });
                                    ASSERT_TRUE(plan.valid);
                                    ASSERT_GT(plan.query_warp_groups, 0);
                                    ASSERT_GT(plan.query_grid_blocks, 0);
                                    if (plan.usesContextParallelism())
                                    {
                                        EXPECT_GT(plan.max_context_partitions, 0);
                                        EXPECT_GT(plan.reducer_dimension_warps, 0);
                                        EXPECT_GT(plan.context_partition_slots, 0);
                                        EXPECT_LE(
                                            plan.context_partition_slots,
                                            plan.max_context_partitions);
                                        EXPECT_GT(plan.partial_output_bytes, 0u);
                                        EXPECT_GT(plan.partial_m_bytes, 0u);
                                        EXPECT_GT(plan.partial_l_bytes, 0u);
                                        if (plan.usesDeviceAdaptiveParallelism())
                                        {
                                            EXPECT_EQ(
                                                axis,
                                                AttentionPrefillParallelAxis::
                                                    GeometrySelected);
                                            EXPECT_GT(
                                                plan.device_direct_partition_limit,
                                                0);
                                            EXPECT_LT(
                                                plan.device_direct_partition_limit,
                                                plan.max_context_partitions);
                                        }
                                        else
                                        {
                                            EXPECT_EQ(
                                                plan.mode,
                                                FA2PrefillPhysicalMode::
                                                    KeyValueContext);
                                            EXPECT_EQ(
                                                plan.device_direct_partition_limit,
                                                0);
                                        }
                                    }
                                    else
                                    {
                                        EXPECT_EQ(
                                            plan.mode,
                                            FA2PrefillPhysicalMode::
                                                QuerySequence);
                                        EXPECT_EQ(
                                            plan.device_direct_partition_limit,
                                            0);
                                        EXPECT_EQ(plan.max_context_partitions, 0);
                                        EXPECT_EQ(plan.context_partition_slots, 0);
                                        EXPECT_EQ(plan.reducer_dimension_warps, 0);
                                        EXPECT_EQ(plan.partial_output_bytes, 0u);
                                        EXPECT_EQ(plan.partial_m_bytes, 0u);
                                        EXPECT_EQ(plan.partial_l_bytes, 0u);
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        EXPECT_FALSE(selectFA2PrefillParallelPlan({}).valid);
        EXPECT_FALSE(selectFA2PrefillParallelPlan({
                         .batch_size = 1,
                         .query_rows = 64,
                         .local_query_heads = 2,
                         .head_dim = 512,
                         .kv_capacity = 131072,
                         .sm_count = 82,
                     }).valid);
        EXPECT_FALSE(selectFA2PrefillParallelPlan({
                         .batch_size = 1,
                         .query_rows = 64,
                         .local_query_heads = 2,
                         .head_dim = 256,
                         .kv_capacity = 0,
                         .sm_count = 82,
                     }).valid);
    }

    TEST(CUDAFlashAttentionLaunchPolicy, ReducerDimensionPolicyIsTotalAndFailClosed)
    {
        constexpr std::array head_dims{32, 64, 128, 256};
        constexpr std::array batches{1, 2, 4};
        constexpr std::array query_rows{1, 2, 15, 16, 17, 64, 128, 512, 4096};
        constexpr std::array local_heads{1, 2, 4, 8, 16, 32, 64};
        constexpr std::array sm_counts{1, 16, 60, 82, 120, 128};

        for (const int head_dim : head_dims)
        {
            const int maximum =
                maximumFA2ReducerDimensionWarps(head_dim);
            ASSERT_GT(maximum, 0);
            for (const int batch : batches)
            {
                for (const int rows : query_rows)
                {
                    for (const int heads : local_heads)
                    {
                        for (const int sms : sm_counts)
                        {
                            const int candidate =
                                selectFA2ReducerDimensionWarps({
                                    .batch_size = batch,
                                    .query_rows = rows,
                                    .local_query_heads = heads,
                                    .head_dim = head_dim,
                                    .sm_count = sms,
                                });
                            ASSERT_GE(candidate, 1);
                            ASSERT_LE(candidate, maximum);
                            EXPECT_EQ(candidate & (candidate - 1), 0);
                        }
                    }
                }
            }
        }

        const FA2QueryPartitionGeometry hd256{
            .batch_size = 1,
            .query_rows = 16,
            .local_query_heads = 2,
            .head_dim = 256,
            .sm_count = 82,
        };
        for (const int candidate : std::array{1, 2, 4, 8})
            EXPECT_EQ(selectFA2ReducerDimensionWarps(hd256, candidate), candidate);
        EXPECT_EQ(selectFA2ReducerDimensionWarps(hd256, 3), 0);
        EXPECT_EQ(selectFA2ReducerDimensionWarps(hd256, -1), 0);

        auto invalid = hd256;
        invalid.query_rows = 0;
        EXPECT_EQ(selectFA2ReducerDimensionWarps(invalid), 0);
        invalid = hd256;
        invalid.head_dim = 512;
        EXPECT_EQ(selectFA2ReducerDimensionWarps(invalid), 0);
        EXPECT_EQ(selectFA2ReducerDimensionWarps({}), 0);
    }

    TEST(CUDAFlashAttentionLaunchPolicy, ValidatesShardedAndReplicatedTPHeadMappings)
    {
        EXPECT_TRUE(isValidFA2HeadMapping({
            .local_query_heads = 4,
            .visible_kv_heads = 1,
            .head_start = 0,
            .replicated_gqa_n_rep = 0,
        }));
        EXPECT_FALSE(isValidFA2HeadMapping({
            .local_query_heads = 3,
            .visible_kv_heads = 2,
            .head_start = 0,
            .replicated_gqa_n_rep = 0,
        }));

        // Qwen 1.5B: 12 global Q heads, two replicated KV heads, TP=4.
        EXPECT_TRUE(isValidFA2HeadMapping({
            .local_query_heads = 3,
            .visible_kv_heads = 2,
            .head_start = 0,
            .replicated_gqa_n_rep = 6,
        }));
        EXPECT_TRUE(isValidFA2HeadMapping({
            .local_query_heads = 3,
            .visible_kv_heads = 2,
            .head_start = 9,
            .replicated_gqa_n_rep = 6,
        }));

        // Qwen 0.8B: one local Q head over replicated 2-KV GQA at TP=8.
        EXPECT_TRUE(isValidFA2HeadMapping({
            .local_query_heads = 1,
            .visible_kv_heads = 2,
            .head_start = 7,
            .replicated_gqa_n_rep = 4,
        }));
        EXPECT_FALSE(isValidFA2HeadMapping({
            .local_query_heads = 3,
            .visible_kv_heads = 2,
            .head_start = 10,
            .replicated_gqa_n_rep = 6,
        }));
        EXPECT_FALSE(isValidFA2HeadMapping({
            .local_query_heads = 1,
            .visible_kv_heads = 1,
            .head_start = -1,
            .replicated_gqa_n_rep = 1,
        }));
    }

    /**
     * @brief Prove HD256 exposes only its economical compiled P@V family.
     *
     * Keeping validation here device-free guarantees capture rejects every
     * non-instantiated value before it reaches CUDA launch dispatch.
     */
    TEST(CUDAFlashAttentionLaunchPolicy, PVWarpGeometryIsTotalAndClosed)
    {
        EXPECT_EQ(maximumFA2PVWarpsPerQueryGroup(64), 1);
        EXPECT_EQ(maximumFA2PVWarpsPerQueryGroup(128), 1);
        EXPECT_EQ(maximumFA2PVWarpsPerQueryGroup(256), 4);
        EXPECT_EQ(maximumFA2PVWarpsPerQueryGroup(0), 0);
        EXPECT_EQ(maximumFA2PVWarpsPerQueryGroup(257), 0);

        for (const int candidate : std::array{1, 2, 4})
            EXPECT_TRUE(isCompiledFA2PVWarpGeometry(256, candidate));
        for (const int rejected : std::array{-1, 0, 3, 6, 8, 16})
            EXPECT_FALSE(isCompiledFA2PVWarpGeometry(256, rejected));

        EXPECT_TRUE(isCompiledFA2PVWarpGeometry(128, 1));
        EXPECT_FALSE(isCompiledFA2PVWarpGeometry(128, 2));
        EXPECT_FALSE(isCompiledFA2PVWarpGeometry(512, 1));
    }

    TEST(CUDAFlashAttentionLaunchPolicy, RejectsInvalidGeometryAndOverrides)
    {
        FA2QueryPartitionGeometry geometry{
            .batch_size = 1,
            .query_rows = 64,
            .local_query_heads = 16,
            .head_dim = 256,
            .sm_count = 82,
        };
        EXPECT_EQ(selectFA2QueryWarpGroups(geometry, 1), 1);
        EXPECT_EQ(selectFA2QueryWarpGroups(geometry, 2), 2);
        EXPECT_EQ(selectFA2QueryWarpGroups(geometry, 3), 0);
        EXPECT_EQ(selectFA2QueryWarpGroups(geometry, -1), 0);

        geometry.query_rows = 0;
        EXPECT_EQ(selectFA2QueryWarpGroups(geometry), 0);
        geometry.query_rows = 64;
        geometry.sm_count = 0;
        EXPECT_EQ(selectFA2QueryWarpGroups(geometry), 0);
        geometry.sm_count = 82;
        geometry.head_dim = 512;
        EXPECT_EQ(selectFA2QueryWarpGroups(geometry), 0);
    }

    TEST(CUDAFlashAttentionLaunchPolicy, SharedMemoryLayoutIdentifiesEveryCompiledTile)
    {
        constexpr std::array head_dims{64, 128, 256};
        constexpr std::array tile_kv_values{16, 32, 64};
        for (const int head_dim : head_dims)
        {
            const int maximum_groups =
                maximumFA2QueryWarpGroups(head_dim);
            for (int groups = 1; groups <= maximum_groups; ++groups)
            {
                std::size_t previous = 0;
                for (const int tile_kv : tile_kv_values)
                {
                    const std::size_t bytes =
                        fa2DynamicSharedMemoryBytes(
                            head_dim,
                            groups,
                            tile_kv);
                    EXPECT_GT(bytes, previous)
                        << "head_dim=" << head_dim
                        << " groups=" << groups
                        << " tile_kv=" << tile_kv;
                    previous = bytes;
                }
            }
        }

        EXPECT_EQ(fa2DynamicSharedMemoryBytes(0, 1, 16), 0u);
        EXPECT_EQ(fa2DynamicSharedMemoryBytes(512, 1, 16), 0u);
        EXPECT_EQ(fa2DynamicSharedMemoryBytes(256, 0, 16), 0u);
        EXPECT_EQ(fa2DynamicSharedMemoryBytes(256, 3, 16), 0u);
        EXPECT_EQ(fa2DynamicSharedMemoryBytes(256, 1, 8), 0u);
        EXPECT_EQ(fa2DynamicSharedMemoryBytes(256, 1, 48), 0u);
    }

    TEST(CUDAFlashAttentionLaunchPolicy, PhysicalKVTilePolicyIsTotalAndFailClosed)
    {
        constexpr std::array head_dims{64, 128, 256};
        constexpr std::array shared_memory_caps{
            std::size_t{48 * 1024},
            std::size_t{64 * 1024},
            std::size_t{100 * 1024},
            std::size_t{164 * 1024},
        };

        for (const int head_dim : head_dims)
        {
            for (int groups = 1;
                 groups <= maximumFA2QueryWarpGroups(head_dim);
                 ++groups)
            {
                for (const std::size_t cap : shared_memory_caps)
                {
                    const FA2KVTileGeometry geometry{
                        .head_dim = head_dim,
                        .query_warp_groups = groups,
                        .max_dynamic_smem = cap,
                    };
                    const int selected = selectFA2KVTile(geometry);
                    const bool tile16_fits =
                        fa2DynamicSharedMemoryBytes(head_dim, groups, 16) <= cap;
                    if (!tile16_fits)
                    {
                        EXPECT_EQ(selected, 0);
                        continue;
                    }

                    const bool wide_hd64 = head_dim == 64 && groups >= 3;
                    const bool tile32_fits =
                        fa2DynamicSharedMemoryBytes(head_dim, groups, 32) <= cap;
                    EXPECT_EQ(selected, wide_hd64 && tile32_fits ? 32 : 16);
                }
            }
        }

        const FA2KVTileGeometry qwen35_ga102{
            .head_dim = 256,
            .query_warp_groups = 2,
            .max_dynamic_smem = 100 * 1024,
        };
        EXPECT_EQ(selectFA2KVTile(qwen35_ga102, 16), 16);
        EXPECT_EQ(selectFA2KVTile(qwen35_ga102, 32), 32);
        EXPECT_EQ(selectFA2KVTile(qwen35_ga102, 64), 0);
        EXPECT_EQ(selectFA2KVTile(qwen35_ga102, 48), 0);
        EXPECT_EQ(selectFA2KVTile(qwen35_ga102, -1), 0);

        EXPECT_EQ(selectFA2KVTile({}), 0);
        EXPECT_EQ(selectFA2KVTile({
                      .head_dim = 256,
                      .query_warp_groups = 3,
                      .max_dynamic_smem = 100 * 1024,
                  }),
                  0);
    }

} // namespace
