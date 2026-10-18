/**
 * @file Test__ROCmFlashAttentionLaunchPolicy.cpp
 * @brief Device-free totality tests for ROCm FA2 capture-time launch policy.
 *
 * The suite covers released Qwen dense and MoE attention geometries, equal TP
 * shards through TP8, unseen positive M values, and K/V capacities through the
 * 256K context regime. Register-score admission also proves exact native-format,
 * visibility, LDS and workspace identity. No HIP runtime or accelerator is used.
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
    namespace policy = llaminar2::rocm::fa2_policy;
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

    /** Exact score ownership depends on admitted geometry, never live KV length. */
    TEST(ROCmFlashAttentionLaunchPolicy, RegisterScoreNativeFormatAndVisibilityTotality)
    {
        for (const int batch : {-1, 0, 1, 2, 4})
        for (const int rows : {0, 1, 2, 4, 8, 16, 31, 128, 255, 511, 512, 513, 1024, 2048, 4096, 65536, 2147483647})
        for (const int width : {0, 64, 128, 192, 255, 256, 257})
        for (const auto format : {llaminar2::TensorType::FP32, llaminar2::TensorType::FP16,
                                 llaminar2::TensorType::BF16, llaminar2::TensorType::Q8_1,
                                 llaminar2::TensorType::TQ8})
        for (const auto visibility : {policy::ROCmFA2VisibilityPolicy::General,
                                     policy::ROCmFA2VisibilityPolicy::CausalFullContext})
        {
            const auto selected = policy::selectROCmFA2QueryScoreStorage({
                .batch_size = batch, .query_rows = rows, .head_dim = width,
                .kv_format = format, .visibility = visibility});
            const bool eligible = batch == 1 && rows >= 512 && width == 256 &&
                format == llaminar2::TensorType::FP16 &&
                visibility == policy::ROCmFA2VisibilityPolicy::CausalFullContext;
            EXPECT_EQ(selected, eligible ? policy::ROCmFA2ScoreStorage::RegisterOwnedXorTile
                                        : policy::ROCmFA2ScoreStorage::SharedMatrix);
        }
    }

    /** A selected layout owns exact storage; invalid requests cannot substitute a tile. */
    TEST(ROCmFlashAttentionLaunchPolicy, RegisterScoreExactLDSRejectsInvalidLayouts)
    {
        constexpr auto storage = policy::ROCmFA2ScoreStorage::RegisterOwnedXorTile;
        constexpr auto query = ROCmFA2PrefillPhysicalMode::QuerySequence;
        const auto tile = selectROCmFA2Tile(256, 16384, 0, query, storage);
        ASSERT_TRUE(tile.valid);
        EXPECT_EQ(tile.lds_bytes, 16384U);
        EXPECT_EQ(tile.query_rows, 16);
        EXPECT_EQ(tile.kv_rows, 16);
        EXPECT_EQ(tile.threads_per_row, 16);
        EXPECT_EQ(tile.score_storage, storage);
        EXPECT_EQ(rocmFA2LDSBytes(256, 16, 16, query, storage), tile.lds_bytes);
        EXPECT_FALSE(selectROCmFA2Tile(256, 16383, 0, query, storage).valid);
        EXPECT_FALSE(selectROCmFA2Tile(128, 65536, 0, query, storage).valid);
        EXPECT_FALSE(selectROCmFA2Tile(256, 65536, 32, query, storage).valid);
        EXPECT_FALSE(selectROCmFA2Tile(256, 65536, 0,
            ROCmFA2PrefillPhysicalMode::KeyValueContext, storage).valid);
        EXPECT_EQ(rocmFA2LDSBytes(256, 16, 32, query, storage), 0U);
        const auto invalid = static_cast<policy::ROCmFA2ScoreStorage>(255);
        EXPECT_EQ(rocmFA2LDSBytes(256, 16, 16, query, invalid), 0U);
        EXPECT_FALSE(selectROCmFA2Tile(256, 65536, 0, query, invalid).valid);
    }

    /** Explicit axes retain ownership; automatic selection uses the measured score layout. */
    TEST(ROCmFlashAttentionLaunchPolicy, RegisterScorePlanPreservesExplicitAxisAndSelectsEconomicalAutomaticAxis)
    {
        for (const int heads : {1, 2, 3, 6, 8, 12, 16, 24})
        for (const int capacity : {512, 8192, 8193, 16384, 65536, 131072, 262144})
        for (const auto axis : {AttentionPrefillParallelAxis::QuerySequence,
                               AttentionPrefillParallelAxis::KeyValueContext,
                               AttentionPrefillParallelAxis::GeometrySelected})
        {
            ROCmFA2PrefillParallelGeometry geometry{
                .batch_size = 1, .query_rows = 512, .local_query_heads = heads,
                .head_dim = 256, .kv_capacity = capacity, .compute_unit_count = 60,
                .lds_capacity_bytes = 65536, .requested_axis = axis};
            const auto canonical = selectROCmFA2PrefillParallelPlan(geometry);
            geometry.kv_format = llaminar2::TensorType::FP16;
            geometry.visibility = policy::ROCmFA2VisibilityPolicy::CausalFullContext;
            const auto selected = selectROCmFA2PrefillParallelPlan(geometry);
            ASSERT_TRUE(canonical.valid);
            ASSERT_TRUE(selected.valid);
            if (axis != AttentionPrefillParallelAxis::GeometrySelected)
                EXPECT_EQ(selected.mode, canonical.mode);
            EXPECT_EQ(selected.query_grid_blocks, canonical.query_grid_blocks);
            EXPECT_LE(selected.partial_output_bytes, canonical.partial_output_bytes);
            EXPECT_LE(selected.partial_m_bytes, canonical.partial_m_bytes);
            EXPECT_LE(selected.partial_l_bytes, canonical.partial_l_bytes);
            if (axis == AttentionPrefillParallelAxis::GeometrySelected && heads >= 6)
                EXPECT_FALSE(selected.usesContextParallelism());
            if (selected.usesContextParallelism())
            {
                EXPECT_EQ(selected.tile.score_storage, policy::ROCmFA2ScoreStorage::SharedMatrix);
                EXPECT_EQ(selected.tile.lds_bytes, canonical.tile.lds_bytes);
            }
            else
            {
                EXPECT_EQ(selected.tile.score_storage, policy::ROCmFA2ScoreStorage::RegisterOwnedXorTile);
                EXPECT_EQ(selected.tile.lds_bytes, 16384U);
                EXPECT_EQ(selected.partial_output_bytes, 0U);
            }
        }
    }

    /** Larger buckets cannot lose the register-score path or escape their exact family BOM. */
    TEST(ROCmFlashAttentionLaunchPolicy, RegisterScoreLargerBucketsAndFamilyEnvelopeAreTotal)
    {
        for (const int heads : {1, 2, 3, 4, 6, 8, 12, 16, 24})
        {
            ROCmFA2PrefillParallelGeometry geometry{
                .batch_size = 1, .query_rows = 4096, .local_query_heads = heads,
                .head_dim = 256, .kv_capacity = 131072, .compute_unit_count = 60,
                .requested_axis = AttentionPrefillParallelAxis::GeometrySelected,
                .kv_format = llaminar2::TensorType::FP16,
                .visibility = policy::ROCmFA2VisibilityPolicy::CausalFullContext};
            const auto family = selectROCmFA2GeometrySelectedWorkspaceEnvelope(geometry);
            ASSERT_TRUE(family.valid);
            std::size_t maximum = 0;
            bool query_selected = false;
            for (int rows = 1; rows <= 4096; ++rows)
            {
                geometry.query_rows = rows;
                const auto plan = selectROCmFA2PrefillParallelPlan(geometry);
                ASSERT_TRUE(plan.valid) << rows << '/' << heads;
                maximum = std::max(maximum, plan.partial_output_bytes);
                EXPECT_LE(plan.partial_output_bytes, family.partial_output_bytes) << rows << '/' << heads;
                if (rows < 512) continue;
                if (query_selected) EXPECT_FALSE(plan.usesContextParallelism()) << rows << '/' << heads;
                if (!plan.usesContextParallelism())
                {
                    query_selected = true;
                    EXPECT_EQ(plan.tile.score_storage, policy::ROCmFA2ScoreStorage::RegisterOwnedXorTile);
                    EXPECT_EQ(plan.tile.lds_bytes, 16384U);
                }
                if (heads >= 6) EXPECT_FALSE(plan.usesContextParallelism());
            }
            EXPECT_EQ(family.partial_output_bytes, maximum) << heads;
        }
    }

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

        const auto invalid_schedule = static_cast<
            llaminar2::rocm::fa2_policy::ROCmFA2PrefillPhysicalMode>(255);
        EXPECT_EQ(rocmFA2LDSBytes(256, 16, 16, invalid_schedule), 0U);
        EXPECT_FALSE(selectROCmFA2Tile(
            256, kROCmFA2LDSCapacityBytes, 0, invalid_schedule).valid);

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
                        EXPECT_EQ(plan.tile.schedule, plan.mode);
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
        EXPECT_EQ(plan.context_phase_block_slots, 180);

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
        const auto tile = selectROCmFA2Tile(geometry.head_dim, kROCmFA2LDSCapacityBytes, 0,
            llaminar2::rocm::fa2_policy::ROCmFA2PrefillPhysicalMode::KeyValueContext);
        ASSERT_TRUE(tile.valid);
        ASSERT_EQ(tile.lds_bytes, 17984U);
        const auto query_tile = selectROCmFA2Tile(geometry.head_dim);
        ASSERT_TRUE(query_tile.valid);
        EXPECT_EQ(query_tile.lds_bytes, 26432U);
        EXPECT_EQ(query_tile.query_rows, tile.query_rows);
        EXPECT_EQ(query_tile.kv_rows, tile.kv_rows);
        ASSERT_LE(tile.lds_bytes, kROCmFA2LDSCapacityBytes / 3);
        ASSERT_GT(tile.lds_bytes, kROCmFA2LDSCapacityBytes / 4);

        // Storage and compiled register envelopes independently cap residency.
        // A smaller LDS budget must bound the physical grid without padding the
        // logical task inventory, and extra LDS cannot admit a fourth group.
        for (const int compute_units : {1, 4, 60, 120})
        for (const int storage_groups : {1, 2, 3, 4})
        {
            auto constrained = geometry;
            constrained.compute_unit_count = compute_units;
            constrained.lds_capacity_bytes = tile.lds_bytes * storage_groups;
            const int hardware = compute_units * (storage_groups < 3 ? storage_groups : 3);
            for (const int slots : {1, 10, 512})
            {
                const auto logical = rocmFA2ContextPhaseLogicalBlocks(constrained, tile, slots);
                EXPECT_EQ(selectROCmFA2ContextPhaseBlockSlots(constrained, tile, slots),
                          logical < hardware ? logical : hardware);
            }
        }

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
            180);
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
            const auto selected = selectROCmFA2Tile(head_dim, kROCmFA2LDSCapacityBytes, 0,
                llaminar2::rocm::fa2_policy::ROCmFA2PrefillPhysicalMode::KeyValueContext);
            ASSERT_TRUE(selected.valid);
            if (head_dim == 256)
                EXPECT_LE(selected.lds_bytes * 3, kROCmFA2LDSCapacityBytes);
            else
                EXPECT_GT(selected.lds_bytes * 2, kROCmFA2LDSCapacityBytes);
        }
    }

    /**
     * @brief Keep every positive production query bucket within real residency.
     *
     * This proves the three-workgroup storage envelope while keeping small
     * arithmetic grids never grow to a padded hardware-capacity grid. The
     * unchanged tile width preserves the canonical softmax arithmetic order.
     */
    TEST(ROCmFlashAttentionLaunchPolicy, WideHeadResidencyAllQueryBucketsAndTP)
    {
        for (const int heads : {16, 24, 32})
        for (const int tp : {1, 2, 4, 8})
        for (int rows = 1; rows <= 512; ++rows)
        for (const int capacity : {256, 257, 4096, 262144})
        {
            const ROCmFA2PrefillParallelGeometry geometry{
                .batch_size = 1,
                .query_rows = rows,
                .local_query_heads = heads / tp,
                .head_dim = 256,
                .kv_capacity = capacity,
                .compute_unit_count = 60,
                .lds_capacity_bytes = kROCmFA2LDSCapacityBytes,
                .requested_axis = AttentionPrefillParallelAxis::KeyValueContext,
            };
            const auto plan = selectROCmFA2PrefillParallelPlan(geometry);
            ASSERT_TRUE(plan.valid);
            ASSERT_EQ(plan.tile.query_rows, 16);
            ASSERT_EQ(plan.tile.kv_rows, 16);
            ASSERT_LE(plan.tile.lds_bytes * 3, kROCmFA2LDSCapacityBytes);
            const auto logical = rocmFA2ContextPhaseLogicalBlocks(
                geometry, plan.tile, plan.context_partition_slots);
            ASSERT_EQ(plan.context_phase_block_slots, logical < 180 ? logical : 180);
            ASSERT_LE(plan.context_phase_block_slots, logical);
        }
    }

} // namespace
